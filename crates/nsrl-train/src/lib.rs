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
pub mod solomon_latent;

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
    MINI_TRANSFORMER_V6_MODEL_MAGIC,
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

const MINI_TRANSFORMER_HOLO_META_DIM: usize = 8;
const MINI_TRANSFORMER_HOLO_ACTION_COUNT: usize = 5;
const MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT: u32 = 8;
const MINI_TRANSFORMER_HOLO_QUERY_SHIFT: u32 = 15;
const MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES: usize = 8;
const MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES: usize = 32;
const MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT: u8 = 2;
const MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT: u8 = 14;
const MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT: u8 = 0;
const MINI_TRANSFORMER_HOLO_ACTION_ATOMS: [[i16; MINI_TRANSFORMER_HOLO_META_DIM];
    MINI_TRANSFORMER_HOLO_ACTION_COUNT] = [
    [16384, -16384, 16384, -16384, 8192, -8192, 4096, -4096],
    [16384, 16384, -16384, -16384, 8192, 8192, -4096, -4096],
    [16384, 0, -16384, 0, 16384, 0, -16384, 0],
    [-16384, 16384, 16384, -16384, -8192, 8192, 4096, -4096],
    [-16384, -16384, -16384, -16384, 8192, 8192, 4096, 4096],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntegerHolographicShiftMemory {
    memory: [[i64; MINI_TRANSFORMER_HOLO_META_DIM]; MINI_TRANSFORMER_HOLO_META_DIM],
    normalizer: [i64; MINI_TRANSFORMER_HOLO_META_DIM],
    update_count: usize,
}

impl IntegerHolographicShiftMemory {
    fn new() -> Self {
        Self {
            memory: [[0_i64; MINI_TRANSFORMER_HOLO_META_DIM]; MINI_TRANSFORMER_HOLO_META_DIM],
            normalizer: [0_i64; MINI_TRANSFORMER_HOLO_META_DIM],
            update_count: 0,
        }
    }

    fn remember(&mut self, state_q15: &[i16; MINI_TRANSFORMER_HOLO_META_DIM], delta: i8) {
        let atom = mini_transformer_holo_action_atom(delta);
        for (row, &atom_value) in atom.iter().enumerate() {
            for (col, &state_value) in state_q15.iter().enumerate() {
                let wide = i64::from(atom_value) * i64::from(state_value);
                self.memory[row][col] = self.memory[row][col]
                    .saturating_add(wide >> MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT);
            }
        }
        for (slot, &value) in self.normalizer.iter_mut().zip(state_q15.iter()) {
            *slot = slot.saturating_add(i64::from(value).abs());
        }
        self.update_count = self.update_count.saturating_add(1);
    }

    fn retrieve_delta(&self, state_q15: &[i16; MINI_TRANSFORMER_HOLO_META_DIM]) -> Option<i8> {
        if self.update_count == 0 {
            return None;
        }
        let mut denominator = 0_i128;
        for (&norm, &state) in self.normalizer.iter().zip(state_q15.iter()) {
            denominator += i128::from(norm) * i128::from(state).abs();
        }
        if denominator == 0 {
            return None;
        }

        let mut recalled = [0_i64; MINI_TRANSFORMER_HOLO_META_DIM];
        for (row, out) in recalled.iter_mut().enumerate() {
            let mut acc = 0_i128;
            for (col, &state) in state_q15.iter().enumerate() {
                acc += i128::from(self.memory[row][col]) * i128::from(state);
            }
            *out = (acc >> MINI_TRANSFORMER_HOLO_QUERY_SHIFT)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        }

        let mut best_delta = 0_i8;
        let mut best_score = i128::MIN;
        for delta in -2_i8..=2 {
            let atom = mini_transformer_holo_action_atom(delta);
            let mut score = 0_i128;
            for (&value, &basis) in recalled.iter().zip(atom.iter()) {
                score += i128::from(value) * i128::from(basis);
            }
            if score > best_score {
                best_score = score;
                best_delta = delta;
            }
        }

        if best_score <= 0 {
            None
        } else {
            Some(best_delta)
        }
    }

    fn hash_into(&self, hasher: &mut StableHasher) {
        hasher.update_usize(self.update_count);
        for row in self.memory {
            for value in row {
                hasher.update_bytes(&value.to_le_bytes());
            }
        }
        for value in self.normalizer {
            hasher.update_bytes(&value.to_le_bytes());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerRuleShiftWindow {
    observation_batches: usize,
    rejected_batches: usize,
    stats: LinearWeightUpdateStats,
}

impl MiniTransformerRuleShiftWindow {
    fn new() -> Self {
        Self {
            observation_batches: 0,
            rejected_batches: 0,
            stats: empty_linear_weight_update_stats(),
        }
    }

    fn observe_accepted(&mut self, stats: LinearWeightUpdateStats) {
        self.observation_batches = self.observation_batches.saturating_add(1);
        self.stats.gradient_saturation_count = self
            .stats
            .gradient_saturation_count
            .saturating_add(stats.gradient_saturation_count);
        self.stats.zero_delta_count = self
            .stats
            .zero_delta_count
            .saturating_add(stats.zero_delta_count);
        self.stats.weight_delta_l1 = self
            .stats
            .weight_delta_l1
            .saturating_add(stats.weight_delta_l1);
    }

    fn observe_rejected(&mut self) {
        self.rejected_batches = self.rejected_batches.saturating_add(1);
        self.observation_batches = self.observation_batches.saturating_add(1);
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerAdaptiveShiftState {
    output_memory: IntegerHolographicShiftMemory,
    mlp_memory: IntegerHolographicShiftMemory,
    embedding_memory: IntegerHolographicShiftMemory,
    q_memory: IntegerHolographicShiftMemory,
    k_memory: IntegerHolographicShiftMemory,
    v_memory: IntegerHolographicShiftMemory,
    o_memory: IntegerHolographicShiftMemory,
    output_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    mlp_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    embedding_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    q_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    k_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    v_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    o_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    output_holo_last_adjust_batch: Option<usize>,
    mlp_holo_last_adjust_batch: Option<usize>,
    embedding_holo_last_adjust_batch: Option<usize>,
    q_holo_last_adjust_batch: Option<usize>,
    k_holo_last_adjust_batch: Option<usize>,
    vo_holo_last_adjust_batch: Option<usize>,
    output_rule: MiniTransformerRuleShiftWindow,
    mlp_rule: MiniTransformerRuleShiftWindow,
    embedding_rule: MiniTransformerRuleShiftWindow,
    q_rule: MiniTransformerRuleShiftWindow,
    k_rule: MiniTransformerRuleShiftWindow,
    v_rule: MiniTransformerRuleShiftWindow,
    o_rule: MiniTransformerRuleShiftWindow,
    output_learning_rate_shift: u8,
    mlp_learning_rate_shift: u8,
    embedding_learning_rate_shift: u8,
    attention_learning_rate_shift: u8,
    attention_q_learning_rate_shift: u8,
    attention_qk_learning_rate_shift: u8,
    adjustment_count: usize,
    rule_adjustment_count: usize,
    rule_update_count: usize,
    rule_event_count: usize,
    holographic_adjustment_count: usize,
}

impl MiniTransformerAdaptiveShiftState {
    fn new(config: MiniTransformerMlpTrainConfig) -> Self {
        Self {
            output_memory: IntegerHolographicShiftMemory::new(),
            mlp_memory: IntegerHolographicShiftMemory::new(),
            embedding_memory: IntegerHolographicShiftMemory::new(),
            q_memory: IntegerHolographicShiftMemory::new(),
            k_memory: IntegerHolographicShiftMemory::new(),
            v_memory: IntegerHolographicShiftMemory::new(),
            o_memory: IntegerHolographicShiftMemory::new(),
            output_previous_state: None,
            mlp_previous_state: None,
            embedding_previous_state: None,
            q_previous_state: None,
            k_previous_state: None,
            v_previous_state: None,
            o_previous_state: None,
            output_holo_last_adjust_batch: None,
            mlp_holo_last_adjust_batch: None,
            embedding_holo_last_adjust_batch: None,
            q_holo_last_adjust_batch: None,
            k_holo_last_adjust_batch: None,
            vo_holo_last_adjust_batch: None,
            output_rule: MiniTransformerRuleShiftWindow::new(),
            mlp_rule: MiniTransformerRuleShiftWindow::new(),
            embedding_rule: MiniTransformerRuleShiftWindow::new(),
            q_rule: MiniTransformerRuleShiftWindow::new(),
            k_rule: MiniTransformerRuleShiftWindow::new(),
            v_rule: MiniTransformerRuleShiftWindow::new(),
            o_rule: MiniTransformerRuleShiftWindow::new(),
            output_learning_rate_shift: config.output_learning_rate_shift,
            mlp_learning_rate_shift: config.mlp_learning_rate_shift,
            embedding_learning_rate_shift: config.embedding_learning_rate_shift,
            attention_learning_rate_shift: config.attention_learning_rate_shift,
            attention_q_learning_rate_shift: config.attention_q_learning_rate_shift,
            attention_qk_learning_rate_shift: config.attention_qk_learning_rate_shift,
            adjustment_count: 0,
            rule_adjustment_count: 0,
            rule_update_count: 0,
            rule_event_count: 0,
            holographic_adjustment_count: 0,
        }
    }

    fn runtime_config(
        &self,
        mut config: MiniTransformerMlpTrainConfig,
    ) -> MiniTransformerMlpTrainConfig {
        if config.adaptive_shift_controller_enabled() {
            config.output_learning_rate_shift = self.output_learning_rate_shift;
            config.mlp_learning_rate_shift = self.mlp_learning_rate_shift;
            config.embedding_learning_rate_shift = self.embedding_learning_rate_shift;
            config.attention_learning_rate_shift = self.attention_learning_rate_shift;
            config.attention_q_learning_rate_shift = self.attention_q_learning_rate_shift;
            config.attention_qk_learning_rate_shift = self.attention_qk_learning_rate_shift;
        }
        config
    }

    #[allow(clippy::too_many_arguments)]
    fn observe_accepted(
        &mut self,
        output: LinearWeightUpdateStats,
        mlp: GatedMlpWeightUpdateStats,
        embedding: SoftmaxUpdateStats,
        attention: &MiniTransformerAttentionWeightUpdateStats,
        accepted_batches: usize,
        enabled: bool,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let rule_enabled = enabled && config.adaptive_rule_shift_controller_enabled();
        let holographic_enabled = enabled && config.adaptive_holographic_shift_controller_enabled();
        if !rule_enabled && !holographic_enabled {
            return;
        }

        let mlp_stats = mini_transformer_gated_mlp_update_stats_as_linear(mlp);
        let embedding_stats = mini_transformer_softmax_update_stats_as_linear(embedding);
        if rule_enabled {
            self.observe_rule_accepted(
                output,
                mlp_stats,
                embedding_stats,
                attention,
                accepted_batches,
                config,
                adaptive_shift_events,
            );
        }
        if !holographic_enabled {
            return;
        }

        let output_state = mini_transformer_holo_shift_state(
            &output,
            mini_transformer_peer_delta_l1(&[mlp_stats, embedding_stats, attention.o]),
            mini_transformer_output_weight_count(),
            false,
            accepted_batches,
        );
        let mlp_state = mini_transformer_holo_shift_state(
            &mlp_stats,
            output.weight_delta_l1,
            mini_transformer_mlp_weight_count(),
            false,
            accepted_batches,
        );
        let embedding_state = mini_transformer_holo_shift_state(
            &embedding_stats,
            output
                .weight_delta_l1
                .saturating_add(mlp_stats.weight_delta_l1),
            mini_transformer_embedding_weight_count(config),
            false,
            accepted_batches,
        );
        let q_state = mini_transformer_holo_shift_state(
            &attention.q,
            attention.k.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let k_state = mini_transformer_holo_shift_state(
            &attention.k,
            attention.q.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let v_state = mini_transformer_holo_shift_state(
            &attention.v,
            attention.o.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let o_state = mini_transformer_holo_shift_state(
            &attention.o,
            attention.v.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );

        let output_teacher = mini_transformer_generic_shift_teacher_delta(
            &output,
            mini_transformer_output_weight_count(),
        );
        let mlp_teacher = mini_transformer_generic_shift_teacher_delta(
            &mlp_stats,
            mini_transformer_mlp_weight_count(),
        );
        let embedding_teacher = mini_transformer_generic_shift_teacher_delta(
            &embedding_stats,
            mini_transformer_embedding_weight_count(config),
        );
        let q_teacher = mini_transformer_attention_q_teacher_delta(attention);
        let k_teacher = mini_transformer_attention_k_teacher_delta(attention);
        let v_teacher = mini_transformer_generic_shift_teacher_delta(
            &attention.v,
            mini_transformer_attention_projection_weight_count(),
        );
        let o_teacher = mini_transformer_generic_shift_teacher_delta(
            &attention.o,
            mini_transformer_attention_projection_weight_count(),
        );

        mini_transformer_holo_remember_lagged(
            &mut self.output_memory,
            &mut self.output_previous_state,
            output_state,
            output_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.mlp_memory,
            &mut self.mlp_previous_state,
            mlp_state,
            mlp_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.embedding_memory,
            &mut self.embedding_previous_state,
            embedding_state,
            embedding_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.q_memory,
            &mut self.q_previous_state,
            q_state,
            q_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.k_memory,
            &mut self.k_previous_state,
            k_state,
            k_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.v_memory,
            &mut self.v_previous_state,
            v_state,
            v_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.o_memory,
            &mut self.o_previous_state,
            o_state,
            o_teacher,
        );

        let output_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                output_teacher,
                self.output_memory
                    .retrieve_delta(&output_state)
                    .unwrap_or(0),
                !rule_enabled,
            ),
            output_teacher,
            self.output_memory.update_count,
            accepted_batches,
            &mut self.output_holo_last_adjust_batch,
        );
        let mlp_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                mlp_teacher,
                self.mlp_memory.retrieve_delta(&mlp_state).unwrap_or(0),
                !rule_enabled,
            ),
            mlp_teacher,
            self.mlp_memory.update_count,
            accepted_batches,
            &mut self.mlp_holo_last_adjust_batch,
        );
        let embedding_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                embedding_teacher,
                self.embedding_memory
                    .retrieve_delta(&embedding_state)
                    .unwrap_or(0),
                !rule_enabled,
            ),
            embedding_teacher,
            self.embedding_memory.update_count,
            accepted_batches,
            &mut self.embedding_holo_last_adjust_batch,
        );
        let q_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                q_teacher,
                self.q_memory.retrieve_delta(&q_state).unwrap_or(0),
                !rule_enabled,
            ),
            q_teacher,
            self.q_memory.update_count,
            accepted_batches,
            &mut self.q_holo_last_adjust_batch,
        );
        let k_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                k_teacher,
                self.k_memory.retrieve_delta(&k_state).unwrap_or(0),
                !rule_enabled,
            ),
            k_teacher,
            self.k_memory.update_count,
            accepted_batches,
            &mut self.k_holo_last_adjust_batch,
        );
        let v_delta = mini_transformer_holo_safety_delta(
            v_teacher,
            self.v_memory.retrieve_delta(&v_state).unwrap_or(0),
            !rule_enabled,
        );
        let o_delta = mini_transformer_holo_safety_delta(
            o_teacher,
            self.o_memory.retrieve_delta(&o_state).unwrap_or(0),
            !rule_enabled,
        );
        let vo_teacher = mini_transformer_join_shift_deltas(v_teacher, o_teacher);
        let vo_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_join_shift_deltas(v_delta, o_delta),
            vo_teacher,
            self.v_memory.update_count.min(self.o_memory.update_count),
            accepted_batches,
            &mut self.vo_holo_last_adjust_batch,
        );

        let adjustment_count_before = self.adjustment_count;
        self.adjust_output(output_delta);
        self.adjust_mlp(mlp_delta);
        self.adjust_embedding(embedding_delta);
        self.adjust_q(q_delta);
        self.adjust_k(k_delta);
        self.adjust_vo(vo_delta);
        self.holographic_adjustment_count = self.holographic_adjustment_count.saturating_add(
            self.adjustment_count
                .saturating_sub(adjustment_count_before),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn observe_rule_accepted(
        &mut self,
        output: LinearWeightUpdateStats,
        mlp: LinearWeightUpdateStats,
        embedding: LinearWeightUpdateStats,
        attention: &MiniTransformerAttentionWeightUpdateStats,
        accepted_batches: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        self.output_rule.observe_accepted(output);
        self.mlp_rule.observe_accepted(mlp);
        self.embedding_rule.observe_accepted(embedding);
        self.q_rule.observe_accepted(attention.q);
        self.k_rule.observe_accepted(attention.k);
        self.v_rule.observe_accepted(attention.v);
        self.o_rule.observe_accepted(attention.o);
        self.apply_rule_controls(accepted_batches, config, adaptive_shift_events);
    }

    fn observe_rule_rejected(
        &mut self,
        rejected_batches: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        self.output_rule.observe_rejected();
        self.mlp_rule.observe_rejected();
        self.embedding_rule.observe_rejected();
        self.q_rule.observe_rejected();
        self.k_rule.observe_rejected();
        self.v_rule.observe_rejected();
        self.o_rule.observe_rejected();
        self.apply_rule_controls(rejected_batches, config, adaptive_shift_events);
    }

    fn apply_rule_controls(
        &mut self,
        batch_index: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let interval = config.adaptive_rule_interval_batches.max(1);
        self.rule_update_count = self.rule_update_count.saturating_add(1);

        let output_rule = self.output_rule;
        self.apply_rule_output(output_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(output_rule, interval) {
            self.output_rule.reset();
        }

        let mlp_rule = self.mlp_rule;
        self.apply_rule_mlp(mlp_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(mlp_rule, interval) {
            self.mlp_rule.reset();
        }

        let embedding_rule = self.embedding_rule;
        self.apply_rule_embedding(
            embedding_rule,
            batch_index,
            interval,
            mini_transformer_embedding_weight_count(config),
            adaptive_shift_events,
        );
        if mini_transformer_rule_should_reset(embedding_rule, interval) {
            self.embedding_rule.reset();
        }

        let q_rule = self.q_rule;
        let k_rule = self.k_rule;
        self.apply_rule_q(q_rule, k_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(q_rule, interval) {
            self.q_rule.reset();
        }

        self.apply_rule_k(k_rule, q_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(k_rule, interval) {
            self.k_rule.reset();
        }

        let v_rule = self.v_rule;
        let o_rule = self.o_rule;
        self.apply_rule_vo(v_rule, o_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(v_rule, interval) {
            self.v_rule.reset();
        }
        if mini_transformer_rule_should_reset(o_rule, interval) {
            self.o_rule.reset();
        }
    }

    fn apply_rule_output(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_generic_delta(
            window,
            mini_transformer_output_weight_count(),
            interval,
        ) {
            let previous = self.output_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "output",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.output_learning_rate_shift = next;
        }
    }

    fn apply_rule_mlp(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_generic_delta(
            window,
            mini_transformer_mlp_weight_count(),
            interval,
        ) {
            let previous = self.mlp_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "mlp",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.mlp_learning_rate_shift = next;
        }
    }

    fn apply_rule_embedding(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        weight_count: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) =
            mini_transformer_rule_generic_delta(window, weight_count, interval)
        {
            let previous = self.embedding_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "embedding",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.embedding_learning_rate_shift = next;
        }
    }

    fn apply_rule_q(
        &mut self,
        q_window: MiniTransformerRuleShiftWindow,
        k_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_q_delta(q_window, k_window, interval) {
            let previous = self.attention_q_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "attention_q",
                    reason,
                    previous,
                    next,
                    delta,
                    q_window,
                ),
            );
            self.attention_q_learning_rate_shift = next;
        }
    }

    fn apply_rule_k(
        &mut self,
        k_window: MiniTransformerRuleShiftWindow,
        q_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_k_delta(k_window, q_window, interval) {
            let previous = self.attention_qk_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "attention_k",
                    reason,
                    previous,
                    next,
                    delta,
                    k_window,
                ),
            );
            self.attention_qk_learning_rate_shift = next;
        }
    }

    fn apply_rule_vo(
        &mut self,
        v_window: MiniTransformerRuleShiftWindow,
        o_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let v_decision = mini_transformer_rule_generic_delta(
            v_window,
            mini_transformer_attention_projection_weight_count(),
            interval,
        );
        let o_decision = mini_transformer_rule_generic_delta(
            o_window,
            mini_transformer_attention_projection_weight_count(),
            interval,
        );
        let Some((delta, reason, window)) =
            mini_transformer_rule_join_vo_decisions(v_decision, o_decision, v_window, o_window)
        else {
            return;
        };
        let previous = self.attention_learning_rate_shift;
        let next = mini_transformer_adjust_shift(previous, delta);
        self.record_rule_event(
            adaptive_shift_events,
            mini_transformer_rule_event(
                batch_index,
                "attention_vo",
                reason,
                previous,
                next,
                delta,
                window,
            ),
        );
        self.attention_learning_rate_shift = next;
    }

    fn record_rule_event(
        &mut self,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
        event: MiniTransformerAdaptiveShiftEventTrace,
    ) {
        if event.previous_shift == event.next_shift {
            return;
        }
        self.rule_adjustment_count = self.rule_adjustment_count.saturating_add(1);
        self.adjustment_count = self.adjustment_count.saturating_add(1);
        self.rule_event_count = self.rule_event_count.saturating_add(1);
        if adaptive_shift_events.len() < MINI_TRANSFORMER_ADAPTIVE_RULE_TRACE_EVENT_LIMIT {
            adaptive_shift_events.push(event);
        }
    }

    fn observe_rejected(
        &mut self,
        rejected_batches: usize,
        enabled: bool,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let rule_enabled = enabled && config.adaptive_rule_shift_controller_enabled();
        let holographic_enabled = enabled && config.adaptive_holographic_shift_controller_enabled();
        if !rule_enabled && !holographic_enabled {
            return;
        }
        if rule_enabled {
            self.observe_rule_rejected(rejected_batches, config, adaptive_shift_events);
        }
        if !holographic_enabled {
            return;
        }
        let output_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_output_weight_count());
        let mlp_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_mlp_weight_count());
        let embedding_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_embedding_weight_count(config));
        let attention_rejected = mini_transformer_rejected_shift_stats(
            mini_transformer_attention_projection_weight_count(),
        );
        let output_state = mini_transformer_holo_shift_state(
            &output_rejected,
            0,
            mini_transformer_output_weight_count(),
            true,
            rejected_batches,
        );
        let mlp_state = mini_transformer_holo_shift_state(
            &mlp_rejected,
            0,
            mini_transformer_mlp_weight_count(),
            true,
            rejected_batches,
        );
        let embedding_state = mini_transformer_holo_shift_state(
            &embedding_rejected,
            0,
            mini_transformer_embedding_weight_count(config),
            true,
            rejected_batches,
        );
        let attention_state = mini_transformer_holo_shift_state(
            &attention_rejected,
            0,
            mini_transformer_attention_projection_weight_count(),
            true,
            rejected_batches,
        );

        mini_transformer_holo_remember_lagged(
            &mut self.output_memory,
            &mut self.output_previous_state,
            output_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.mlp_memory,
            &mut self.mlp_previous_state,
            mlp_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.embedding_memory,
            &mut self.embedding_previous_state,
            embedding_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.q_memory,
            &mut self.q_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.k_memory,
            &mut self.k_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.v_memory,
            &mut self.v_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.o_memory,
            &mut self.o_previous_state,
            attention_state,
            1,
        );
        let adjustment_count_before = self.adjustment_count;
        self.adjust_output(1);
        self.adjust_mlp(1);
        self.adjust_embedding(1);
        self.adjust_q(1);
        self.adjust_k(1);
        self.adjust_vo(1);
        self.holographic_adjustment_count = self.holographic_adjustment_count.saturating_add(
            self.adjustment_count
                .saturating_sub(adjustment_count_before),
        );
    }

    fn total_memory_updates(&self) -> usize {
        self.output_memory
            .update_count
            .saturating_add(self.mlp_memory.update_count)
            .saturating_add(self.embedding_memory.update_count)
            .saturating_add(self.q_memory.update_count)
            .saturating_add(self.k_memory.update_count)
            .saturating_add(self.v_memory.update_count)
            .saturating_add(self.o_memory.update_count)
    }

    fn attention_memory_updates(&self) -> usize {
        self.q_memory
            .update_count
            .saturating_add(self.k_memory.update_count)
            .saturating_add(self.v_memory.update_count)
            .saturating_add(self.o_memory.update_count)
    }

    fn memory_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(usize::from(self.output_learning_rate_shift));
        hasher.update_usize(usize::from(self.mlp_learning_rate_shift));
        hasher.update_usize(usize::from(self.embedding_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_q_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_qk_learning_rate_shift));
        self.output_memory.hash_into(&mut hasher);
        self.mlp_memory.hash_into(&mut hasher);
        self.embedding_memory.hash_into(&mut hasher);
        self.q_memory.hash_into(&mut hasher);
        self.k_memory.hash_into(&mut hasher);
        self.v_memory.hash_into(&mut hasher);
        self.o_memory.hash_into(&mut hasher);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.output_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.mlp_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.embedding_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.q_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.k_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.v_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.o_previous_state);
        mini_transformer_hash_optional_usize(&mut hasher, self.output_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.mlp_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.embedding_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.q_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.k_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.vo_holo_last_adjust_batch);
        hasher.finish()
    }

    fn attention_memory_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(usize::from(self.attention_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_q_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_qk_learning_rate_shift));
        self.q_memory.hash_into(&mut hasher);
        self.k_memory.hash_into(&mut hasher);
        self.v_memory.hash_into(&mut hasher);
        self.o_memory.hash_into(&mut hasher);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.q_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.k_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.v_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.o_previous_state);
        mini_transformer_hash_optional_usize(&mut hasher, self.q_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.k_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.vo_holo_last_adjust_batch);
        hasher.finish()
    }

    fn adjust_output(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.output_learning_rate_shift, delta);
        if next != self.output_learning_rate_shift {
            self.output_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    fn adjust_mlp(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.mlp_learning_rate_shift, delta);
        if next != self.mlp_learning_rate_shift {
            self.mlp_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    fn adjust_embedding(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.embedding_learning_rate_shift, delta);
        if next != self.embedding_learning_rate_shift {
            self.embedding_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    fn adjust_q(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_q_learning_rate_shift, delta);
        if next != self.attention_q_learning_rate_shift {
            self.attention_q_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    fn adjust_k(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_qk_learning_rate_shift, delta);
        if next != self.attention_qk_learning_rate_shift {
            self.attention_qk_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    fn adjust_vo(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_learning_rate_shift, delta);
        if next != self.attention_learning_rate_shift {
            self.attention_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }
}

fn mini_transformer_holo_action_atom(delta: i8) -> &'static [i16; MINI_TRANSFORMER_HOLO_META_DIM] {
    match delta.clamp(-2, 2) {
        -2 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[0],
        -1 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[1],
        0 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[2],
        1 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[3],
        _ => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[4],
    }
}

fn mini_transformer_adjust_shift(current: u8, delta: i8) -> u8 {
    let next = i16::from(current) + i16::from(delta);
    next.clamp(0, i16::from(MAX_RIGHT_SHIFT)) as u8
}

fn mini_transformer_rule_generic_delta(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if window.observation_batches < interval.max(1) {
        return None;
    }
    if mini_transformer_rule_saturation_pressure(window, weight_count) {
        return Some((1, "saturation"));
    }
    if mini_transformer_rule_zero_pressure(window, weight_count) {
        return Some((-1, "zero_delta"));
    }
    None
}

fn mini_transformer_rule_q_delta(
    q_window: MiniTransformerRuleShiftWindow,
    k_window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if q_window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if q_window.observation_batches < interval.max(1) {
        return None;
    }
    if q_window.stats.weight_delta_l1 == 0
        || mini_transformer_rule_zero_pressure(
            q_window,
            mini_transformer_attention_projection_weight_count(),
        )
    {
        return Some((-1, "zero_delta"));
    }
    if k_window.stats.weight_delta_l1 > 0
        && q_window.stats.weight_delta_l1.saturating_mul(8) < k_window.stats.weight_delta_l1
    {
        return Some((-1, "lagging_k"));
    }
    if mini_transformer_rule_saturation_pressure(
        q_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((1, "saturation"));
    }
    if k_window.stats.weight_delta_l1 > 0
        && q_window.stats.weight_delta_l1 > k_window.stats.weight_delta_l1.saturating_mul(4)
    {
        return Some((1, "overpowering_k"));
    }
    None
}

fn mini_transformer_rule_k_delta(
    k_window: MiniTransformerRuleShiftWindow,
    q_window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if k_window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if k_window.observation_batches < interval.max(1) {
        return None;
    }
    if mini_transformer_rule_zero_pressure(
        k_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((-1, "zero_delta"));
    }
    if mini_transformer_rule_saturation_pressure(
        k_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((1, "saturation"));
    }
    if q_window.stats.weight_delta_l1 > 0
        && k_window.stats.weight_delta_l1 > q_window.stats.weight_delta_l1.saturating_mul(64)
    {
        return Some((1, "overpowering_q"));
    }
    None
}

fn mini_transformer_rule_saturation_pressure(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
) -> bool {
    if window.stats.gradient_saturation_count == 0 {
        return false;
    }
    let total_slots = weight_count
        .max(1)
        .saturating_mul(window.observation_batches.max(1));
    let threshold = (total_slots / MINI_TRANSFORMER_RULE_SATURATION_PRESSURE_DIVISOR)
        .max(window.observation_batches.max(1));
    window.stats.gradient_saturation_count >= threshold
}

fn mini_transformer_rule_join_vo_decisions(
    v_decision: Option<(i8, &'static str)>,
    o_decision: Option<(i8, &'static str)>,
    v_window: MiniTransformerRuleShiftWindow,
    o_window: MiniTransformerRuleShiftWindow,
) -> Option<(i8, &'static str, MiniTransformerRuleShiftWindow)> {
    let (delta, reason) = match (v_decision, o_decision) {
        (Some((v_delta, v_reason)), Some((o_delta, o_reason))) => {
            let delta = mini_transformer_join_shift_deltas(v_delta, o_delta);
            let reason = if v_reason == o_reason || delta == v_delta {
                v_reason
            } else {
                o_reason
            };
            (delta, reason)
        }
        (Some(decision), None) | (None, Some(decision)) => decision,
        (None, None) => return None,
    };
    Some((
        delta,
        reason,
        mini_transformer_rule_join_windows(v_window, o_window),
    ))
}

fn mini_transformer_rule_join_windows(
    left: MiniTransformerRuleShiftWindow,
    right: MiniTransformerRuleShiftWindow,
) -> MiniTransformerRuleShiftWindow {
    MiniTransformerRuleShiftWindow {
        observation_batches: left.observation_batches.max(right.observation_batches),
        rejected_batches: left.rejected_batches.saturating_add(right.rejected_batches),
        stats: LinearWeightUpdateStats {
            gradient_saturation_count: left
                .stats
                .gradient_saturation_count
                .saturating_add(right.stats.gradient_saturation_count),
            zero_delta_count: left
                .stats
                .zero_delta_count
                .saturating_add(right.stats.zero_delta_count),
            weight_delta_l1: left
                .stats
                .weight_delta_l1
                .saturating_add(right.stats.weight_delta_l1),
        },
    }
}

fn mini_transformer_rule_zero_pressure(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
) -> bool {
    if window.stats.weight_delta_l1 == 0 {
        return true;
    }
    let total_slots = weight_count
        .max(1)
        .saturating_mul(window.observation_batches.max(1));
    let zero_pressure_threshold =
        ((total_slots as u128) * (MINI_TRANSFORMER_RULE_ZERO_PRESSURE_NUMERATOR as u128)
            / (MINI_TRANSFORMER_RULE_ZERO_PRESSURE_DENOMINATOR as u128)) as usize;
    window.stats.zero_delta_count > zero_pressure_threshold
}

fn mini_transformer_rule_should_reset(
    window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> bool {
    window.rejected_batches > 0 || window.observation_batches >= interval.max(1)
}

fn mini_transformer_rule_event(
    batch_index: usize,
    component: &'static str,
    reason: &'static str,
    previous_shift: u8,
    next_shift: u8,
    delta: i8,
    window: MiniTransformerRuleShiftWindow,
) -> MiniTransformerAdaptiveShiftEventTrace {
    MiniTransformerAdaptiveShiftEventTrace {
        batch_index,
        component,
        reason,
        previous_shift,
        next_shift,
        delta,
        observation_batches: window.observation_batches,
        rejected_batches: window.rejected_batches,
        saturation_count: window.stats.gradient_saturation_count,
        zero_delta_count: window.stats.zero_delta_count,
        weight_delta_l1: window.stats.weight_delta_l1,
    }
}

fn mini_transformer_output_weight_count() -> usize {
    BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
}

fn mini_transformer_mlp_weight_count() -> usize {
    MINI_TRANSFORMER_D_MODEL
        .saturating_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .saturating_mul(2)
        .saturating_add(MINI_TRANSFORMER_HIDDEN_DIM.saturating_mul(MINI_TRANSFORMER_D_MODEL))
}

fn mini_transformer_attention_projection_weight_count() -> usize {
    MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
}

fn mini_transformer_embedding_weight_count(config: MiniTransformerMlpTrainConfig) -> usize {
    let token_embeddings = BYTE_VOCAB.saturating_mul(MINI_TRANSFORMER_D_MODEL);
    if config.position_policy.uses_position_embeddings() {
        token_embeddings.saturating_add(config.seq_len.saturating_mul(MINI_TRANSFORMER_D_MODEL))
    } else {
        token_embeddings
    }
}

fn mini_transformer_peer_delta_l1(stats: &[LinearWeightUpdateStats]) -> u64 {
    stats.iter().fold(0_u64, |total, stats| {
        total.saturating_add(stats.weight_delta_l1)
    })
}

fn mini_transformer_softmax_update_stats_as_linear(
    stats: SoftmaxUpdateStats,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: stats.gradient_saturation_count,
        zero_delta_count: stats.zero_delta_count,
        weight_delta_l1: stats.weight_delta_l1,
    }
}

fn mini_transformer_gated_mlp_update_stats_as_linear(
    stats: GatedMlpWeightUpdateStats,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: stats.gradient_saturation_count().unwrap_or(usize::MAX),
        zero_delta_count: stats.zero_delta_count().unwrap_or(usize::MAX),
        weight_delta_l1: stats.weight_delta_l1().unwrap_or(u64::MAX),
    }
}

fn mini_transformer_generic_shift_teacher_delta(
    stats: &LinearWeightUpdateStats,
    weight_count: usize,
) -> i8 {
    if stats.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.weight_delta_l1 == 0 || stats.zero_delta_count > weight_count.max(1) / 2 {
        return -1;
    }
    0
}

fn mini_transformer_holo_remember_lagged(
    memory: &mut IntegerHolographicShiftMemory,
    previous_state: &mut Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    current_state: [i16; MINI_TRANSFORMER_HOLO_META_DIM],
    teacher: i8,
) {
    if let Some(state) = previous_state {
        memory.remember(state, teacher);
    }
    *previous_state = Some(current_state);
}

fn mini_transformer_hash_holo_previous_state(
    hasher: &mut StableHasher,
    previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
) {
    match previous_state {
        Some(state) => {
            hasher.update_usize(1);
            for value in state {
                hasher.update_bytes(&value.to_le_bytes());
            }
        }
        None => hasher.update_usize(0),
    }
}

fn mini_transformer_hash_optional_usize(hasher: &mut StableHasher, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update_usize(1);
            hasher.update_usize(value);
        }
        None => hasher.update_usize(0),
    }
}

fn mini_transformer_rejected_shift_stats(weight_count: usize) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: 1,
        zero_delta_count: weight_count,
        weight_delta_l1: 0,
    }
}

fn mini_transformer_join_shift_deltas(left: i8, right: i8) -> i8 {
    if left > 0 || right > 0 {
        left.max(right)
    } else {
        left.min(right)
    }
}

fn mini_transformer_holo_safety_delta(teacher: i8, recalled: i8, teacher_can_act: bool) -> i8 {
    if teacher_can_act && teacher != 0 {
        teacher
    } else if teacher == 0 {
        recalled.clamp(-1, 1)
    } else {
        0
    }
}

fn mini_transformer_holo_authorized_delta(
    candidate: i8,
    teacher: i8,
    memory_update_count: usize,
    batch_index: usize,
    last_adjust_batch: &mut Option<usize>,
) -> i8 {
    if candidate == 0 {
        return 0;
    }
    if teacher == 0 && memory_update_count < MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES {
        return 0;
    }
    if let Some(last_batch) = *last_adjust_batch
        && batch_index.saturating_sub(last_batch)
            < MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES
    {
        return 0;
    }
    *last_adjust_batch = Some(batch_index);
    candidate
}

fn mini_transformer_holo_shift_state(
    stats: &LinearWeightUpdateStats,
    peer_delta_l1: u64,
    weight_count: usize,
    rejected: bool,
    phase: usize,
) -> [i16; MINI_TRANSFORMER_HOLO_META_DIM] {
    let movement = mini_transformer_log_u64_q15(stats.weight_delta_l1);
    let peer_movement = mini_transformer_log_u64_q15(peer_delta_l1);
    let zero_ratio = mini_transformer_ratio_q15(stats.zero_delta_count, weight_count);
    let saturation_ratio =
        mini_transformer_ratio_q15(stats.gradient_saturation_count, weight_count.max(1));
    let phase_q15 = mini_transformer_ratio_q15(phase.min(1024), 1024);
    let rejected_q15 = if rejected { i16::MAX } else { 0 };
    let signed_pressure = if stats.gradient_saturation_count > 0 || rejected {
        i16::MAX
    } else if stats.weight_delta_l1 == 0 || stats.zero_delta_count > weight_count / 2 {
        -i16::MAX
    } else {
        0
    };

    [
        i16::MAX,
        movement,
        peer_movement,
        zero_ratio,
        saturation_ratio,
        rejected_q15,
        phase_q15,
        signed_pressure,
    ]
}

fn mini_transformer_log_u64_q15(value: u64) -> i16 {
    if value == 0 {
        return 0;
    }
    let bits = 64_u32.saturating_sub(value.leading_zeros());
    let scaled = bits.saturating_mul(512).min(i16::MAX as u32);
    scaled as i16
}

fn mini_transformer_ratio_q15(numerator: usize, denominator: usize) -> i16 {
    if denominator == 0 {
        return 0;
    }
    let wide = (numerator as u128).saturating_mul(i16::MAX as u128) / (denominator as u128);
    wide.min(i16::MAX as u128) as i16
}

const MINI_TRANSFORMER_TRACE_SUMMARY_INITIAL_STEPS: usize = 16;
const MINI_TRANSFORMER_TRACE_SUMMARY_DEFAULT_INTERVAL_STEPS: usize = 1024;

fn mini_transformer_trace_sample_interval(
    progress_interval_batches: usize,
    batch_windows: usize,
) -> usize {
    let progress_windows = progress_interval_batches.saturating_mul(batch_windows);
    if progress_windows == 0 {
        MINI_TRANSFORMER_TRACE_SUMMARY_DEFAULT_INTERVAL_STEPS
    } else {
        progress_windows
    }
}

fn mini_transformer_should_record_step(
    trace_detail: MiniTransformerTraceDetail,
    update_index: usize,
    sample_interval: usize,
) -> bool {
    match trace_detail {
        MiniTransformerTraceDetail::Full => true,
        MiniTransformerTraceDetail::Summary => {
            update_index <= MINI_TRANSFORMER_TRACE_SUMMARY_INITIAL_STEPS
                || update_index.is_multiple_of(sample_interval.max(1))
        }
        MiniTransformerTraceDetail::None => false,
    }
}

fn emit_mini_transformer_committed_binary_steps<G>(
    steps: &[MiniTransformerMlpTrainingStepTrace],
    start_index: usize,
    binary_trace: &mut G,
) -> Result<(), TrainError>
where
    G: FnMut(MiniTransformerBinaryTraceRecord<'_>) -> Result<(), TrainError>,
{
    for step in &steps[start_index.min(steps.len())..] {
        binary_trace(MiniTransformerBinaryTraceRecord::StepSample(step))?;
    }
    Ok(())
}

fn mini_transformer_attention_q_teacher_delta(
    stats: &MiniTransformerAttentionWeightUpdateStats,
) -> i8 {
    if stats.q.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.q.weight_delta_l1 == 0 {
        return -1;
    }
    if stats.k.weight_delta_l1 > 0
        && stats.q.weight_delta_l1.saturating_mul(8) < stats.k.weight_delta_l1
    {
        return -1;
    }
    if stats.k.weight_delta_l1 > 0
        && stats.q.weight_delta_l1 > stats.k.weight_delta_l1.saturating_mul(4)
    {
        return 1;
    }
    0
}

fn mini_transformer_attention_k_teacher_delta(
    stats: &MiniTransformerAttentionWeightUpdateStats,
) -> i8 {
    if stats.k.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.k.weight_delta_l1 == 0 {
        return -1;
    }
    if stats.q.weight_delta_l1 > 0
        && stats.k.weight_delta_l1 > stats.q.weight_delta_l1.saturating_mul(64)
    {
        return 1;
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_training_progress_trace(
    config: MiniTransformerMlpTrainConfig,
    token_count: usize,
    token_hash: u64,
    window_hash: u64,
    windows: usize,
    examined_windows: usize,
    updates: usize,
    accepted_batch_count: usize,
    rejected_batch_count: usize,
    rollback_count: usize,
    rejected_window_count: usize,
    output_head_delta_l1: u64,
    mlp_delta_l1: u64,
    embedding_delta_l1: u64,
    attention_delta_l1: u64,
    attention_q_delta_l1: u64,
    attention_k_delta_l1: u64,
    attention_v_delta_l1: u64,
    attention_o_delta_l1: u64,
    output_head_carry_l1: u64,
    mlp_carry_l1: u64,
    embedding_carry_l1: u64,
    attention_carry_l1: u64,
    attention_q_carry_l1: u64,
    attention_k_carry_l1: u64,
    attention_v_carry_l1: u64,
    attention_o_carry_l1: u64,
    adaptive_attention_shifts: &MiniTransformerAdaptiveShiftState,
    model: &MiniTransformerMlpModel,
) -> MiniTransformerMlpTrainingProgressTrace {
    let runtime_config = adaptive_attention_shifts.runtime_config(config);
    MiniTransformerMlpTrainingProgressTrace {
        config,
        token_count,
        token_hash,
        window_hash,
        windows,
        examined_windows,
        updates,
        accepted_batch_count,
        rejected_batch_count,
        rollback_count,
        rejected_window_count,
        output_head_delta_l1,
        mlp_delta_l1,
        embedding_delta_l1,
        attention_delta_l1,
        attention_q_delta_l1,
        attention_k_delta_l1,
        attention_v_delta_l1,
        attention_o_delta_l1,
        output_head_carry_l1,
        mlp_carry_l1,
        embedding_carry_l1,
        attention_carry_l1,
        attention_q_carry_l1,
        attention_k_carry_l1,
        attention_v_carry_l1,
        attention_o_carry_l1,
        adaptive_rule_shift_adjustment_count: adaptive_attention_shifts.rule_adjustment_count,
        adaptive_holographic_shift_adjustment_count: adaptive_attention_shifts
            .holographic_adjustment_count,
        current_output_learning_rate_shift: runtime_config.output_learning_rate_shift,
        current_mlp_learning_rate_shift: runtime_config.mlp_learning_rate_shift,
        current_embedding_learning_rate_shift: runtime_config.embedding_learning_rate_shift,
        current_attention_learning_rate_shift: runtime_config.attention_learning_rate_shift,
        current_attention_q_learning_rate_shift: runtime_config.attention_q_learning_rate_shift,
        current_attention_qk_learning_rate_shift: runtime_config.attention_qk_learning_rate_shift,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
    }
}

struct MiniTransformerHostTrainCoreWorkspaceBuffers {
    embedding_output: Vec<i16>,
    attention_norm: Vec<i16>,
    attention_q: Vec<i16>,
    attention_k: Vec<i16>,
    attention_v: Vec<i16>,
    attention_context: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    attention_state_kv: Vec<i64>,
    attention_key_sums: Vec<i64>,
    mlp_norm: Vec<i16>,
    mlp_up: Vec<i16>,
    mlp_gate: Vec<i16>,
    mlp_gated: Vec<i16>,
    mlp_output: Vec<i16>,
    block_output: Vec<i16>,
    logits_q8: Vec<i32>,
    probabilities_q15: Vec<i16>,
    grad_output_q15: Vec<i16>,
    output_scaled_grad: Vec<i32>,
    grad_last_features: Vec<i16>,
    grad_mlp_output: Vec<i16>,
    grad_mlp_input: Vec<i16>,
    mlp_scaled_grad: Vec<i32>,
    mlp_input_grad_gated: Vec<i16>,
    mlp_input_grad_up: Vec<i16>,
    mlp_input_grad_gate: Vec<i16>,
    mlp_input_grad_up_input: Vec<i16>,
    mlp_input_grad_gate_input: Vec<i16>,
    mlp_update_grad_gated: Vec<i16>,
    mlp_update_grad_up: Vec<i16>,
    mlp_update_grad_gate: Vec<i16>,
    grad_attention_output: Vec<i16>,
    grad_attention_context: Vec<i16>,
    attention_scaled_grad: Vec<i32>,
    linear_prefix_states: Vec<i64>,
    linear_denominators: Vec<i64>,
    linear_grad_state_q15: Vec<i64>,
    linear_grad_q_acc: Vec<i64>,
    linear_grad_k_acc: Vec<i64>,
    linear_grad_v_acc: Vec<i64>,
    grad_attention_q: Vec<i16>,
    grad_attention_k: Vec<i16>,
    grad_attention_v: Vec<i16>,
    grad_attention_norm_input: Vec<i16>,
    grad_embedding_output: Vec<i16>,
}

impl MiniTransformerHostTrainCoreWorkspaceBuffers {
    fn new(seq_len: usize) -> Result<Self, TrainError> {
        let total = seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidConfig)?;
        let hidden_total = seq_len
            .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
            .ok_or(TrainError::InvalidConfig)?;
        let head_dim = mini_transformer_head_dim()?;
        let head_state_len = head_dim
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let state_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_len = seq_len
            .checked_mul(state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let denom_len = seq_len
            .checked_mul(MINI_TRANSFORMER_HEADS)
            .ok_or(TrainError::InvalidConfig)?;
        let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

        Ok(Self {
            embedding_output: vec![0_i16; total],
            attention_norm: vec![0_i16; total],
            attention_q: vec![0_i16; total],
            attention_k: vec![0_i16; total],
            attention_v: vec![0_i16; total],
            attention_context: vec![0_i16; total],
            attention_output: vec![0_i16; total],
            attention_residual: vec![0_i16; total],
            attention_state_kv: vec![0_i64; state_len],
            attention_key_sums: vec![0_i64; key_sum_len],
            mlp_norm: vec![0_i16; total],
            mlp_up: vec![0_i16; hidden_total],
            mlp_gate: vec![0_i16; hidden_total],
            mlp_gated: vec![0_i16; hidden_total],
            mlp_output: vec![0_i16; total],
            block_output: vec![0_i16; total],
            logits_q8: vec![0_i32; BYTE_VOCAB],
            probabilities_q15: vec![0_i16; BYTE_VOCAB],
            grad_output_q15: vec![0_i16; BYTE_VOCAB],
            output_scaled_grad: vec![0_i32; BYTE_VOCAB],
            grad_last_features: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
            grad_mlp_output: vec![0_i16; total],
            grad_mlp_input: vec![0_i16; total],
            mlp_scaled_grad: vec![0_i32; scaled_len],
            mlp_input_grad_gated: vec![0_i16; hidden_total],
            mlp_input_grad_up: vec![0_i16; hidden_total],
            mlp_input_grad_gate: vec![0_i16; hidden_total],
            mlp_input_grad_up_input: vec![0_i16; total],
            mlp_input_grad_gate_input: vec![0_i16; total],
            mlp_update_grad_gated: vec![0_i16; hidden_total],
            mlp_update_grad_up: vec![0_i16; hidden_total],
            mlp_update_grad_gate: vec![0_i16; hidden_total],
            grad_attention_output: vec![0_i16; total],
            grad_attention_context: vec![0_i16; total],
            attention_scaled_grad: vec![0_i32; MINI_TRANSFORMER_D_MODEL],
            linear_prefix_states: vec![0_i64; prefix_len],
            linear_denominators: vec![0_i64; denom_len],
            linear_grad_state_q15: vec![0_i64; head_state_len],
            linear_grad_q_acc: vec![0_i64; total],
            linear_grad_k_acc: vec![0_i64; total],
            linear_grad_v_acc: vec![0_i64; total],
            grad_attention_q: vec![0_i16; total],
            grad_attention_k: vec![0_i16; total],
            grad_attention_v: vec![0_i16; total],
            grad_attention_norm_input: vec![0_i16; total],
            grad_embedding_output: vec![0_i16; total],
        })
    }

    fn as_workspace(&mut self) -> nsrl_train_core::MiniTransformerStepWorkspace<'_> {
        nsrl_train_core::MiniTransformerStepWorkspace {
            embedding_output: &mut self.embedding_output,
            attention_norm: &mut self.attention_norm,
            attention_q: &mut self.attention_q,
            attention_k: &mut self.attention_k,
            attention_v: &mut self.attention_v,
            attention_context: &mut self.attention_context,
            attention_output: &mut self.attention_output,
            attention_residual: &mut self.attention_residual,
            attention_state_kv: &mut self.attention_state_kv,
            attention_key_sums: &mut self.attention_key_sums,
            mlp_norm: &mut self.mlp_norm,
            mlp_up: &mut self.mlp_up,
            mlp_gate: &mut self.mlp_gate,
            mlp_gated: &mut self.mlp_gated,
            mlp_output: &mut self.mlp_output,
            block_output: &mut self.block_output,
            logits_q8: &mut self.logits_q8,
            probabilities_q15: &mut self.probabilities_q15,
            grad_output_q15: &mut self.grad_output_q15,
            output_scaled_grad: &mut self.output_scaled_grad,
            grad_last_features: &mut self.grad_last_features,
            grad_mlp_output: &mut self.grad_mlp_output,
            grad_mlp_input: &mut self.grad_mlp_input,
            mlp_scaled_grad: &mut self.mlp_scaled_grad,
            mlp_input_grad_gated: &mut self.mlp_input_grad_gated,
            mlp_input_grad_up: &mut self.mlp_input_grad_up,
            mlp_input_grad_gate: &mut self.mlp_input_grad_gate,
            mlp_input_grad_up_input: &mut self.mlp_input_grad_up_input,
            mlp_input_grad_gate_input: &mut self.mlp_input_grad_gate_input,
            mlp_update_grad_gated: &mut self.mlp_update_grad_gated,
            mlp_update_grad_up: &mut self.mlp_update_grad_up,
            mlp_update_grad_gate: &mut self.mlp_update_grad_gate,
            grad_attention_output: &mut self.grad_attention_output,
            grad_attention_context: &mut self.grad_attention_context,
            attention_scaled_grad: &mut self.attention_scaled_grad,
            linear_prefix_states: &mut self.linear_prefix_states,
            linear_denominators: &mut self.linear_denominators,
            linear_grad_state_q15: &mut self.linear_grad_state_q15,
            linear_grad_q_acc: &mut self.linear_grad_q_acc,
            linear_grad_k_acc: &mut self.linear_grad_k_acc,
            linear_grad_v_acc: &mut self.linear_grad_v_acc,
            grad_attention_q: &mut self.grad_attention_q,
            grad_attention_k: &mut self.grad_attention_k,
            grad_attention_v: &mut self.grad_attention_v,
            grad_attention_norm_input: &mut self.grad_attention_norm_input,
            grad_embedding_output: &mut self.grad_embedding_output,
        }
    }

    fn reset_host_training_step(&mut self) {
        self.grad_mlp_output.fill(0);
    }

    fn validate_host_training_step_shape(&self, seq_len: usize) -> Result<(), TrainError> {
        let total = seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidConfig)?;
        let hidden_total = seq_len
            .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
            .ok_or(TrainError::InvalidConfig)?;
        let head_dim = mini_transformer_head_dim()?;
        let head_state_len = head_dim
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let state_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_len = seq_len
            .checked_mul(state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let denom_len = seq_len
            .checked_mul(MINI_TRANSFORMER_HEADS)
            .ok_or(TrainError::InvalidConfig)?;
        let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

        if self.output_scaled_grad.len() != BYTE_VOCAB
            || self.grad_last_features.len() != MINI_TRANSFORMER_D_MODEL
            || self.grad_mlp_output.len() != total
            || self.grad_mlp_input.len() != total
            || self.mlp_scaled_grad.len() != scaled_len
            || self.mlp_input_grad_gated.len() != hidden_total
            || self.mlp_input_grad_up.len() != hidden_total
            || self.mlp_input_grad_gate.len() != hidden_total
            || self.mlp_input_grad_up_input.len() != total
            || self.mlp_input_grad_gate_input.len() != total
            || self.mlp_update_grad_gated.len() != hidden_total
            || self.mlp_update_grad_up.len() != hidden_total
            || self.mlp_update_grad_gate.len() != hidden_total
            || self.grad_attention_output.len() != total
            || self.grad_attention_context.len() != total
            || self.attention_scaled_grad.len() < MINI_TRANSFORMER_D_MODEL
            || self.attention_state_kv.len() != state_len
            || self.attention_key_sums.len() != key_sum_len
            || self.linear_prefix_states.len() != prefix_len
            || self.linear_denominators.len() != denom_len
            || self.linear_grad_state_q15.len() != head_state_len
            || self.linear_grad_q_acc.len() != total
            || self.linear_grad_k_acc.len() != total
            || self.linear_grad_v_acc.len() != total
            || self.grad_attention_q.len() != total
            || self.grad_attention_k.len() != total
            || self.grad_attention_v.len() != total
            || self.grad_attention_norm_input.len() != total
            || self.grad_embedding_output.len() != total
        {
            return Err(TrainError::InvalidConfig);
        }
        Ok(())
    }
}

fn mini_transformer_uses_train_core_step(config: MiniTransformerMlpTrainConfig) -> bool {
    config.batch_windows == 1
        && config.tokenizer_id == ByteTokenizerId::Identity
        && config.attention_kind == MiniTransformerAttentionKind::Linear
        && config.position_policy == MiniTransformerPositionPolicy::Nope
        && !config.adaptive_shift_controller_enabled()
        && !config.attention_vo_error_feedback
        && !config.attention_vo_oracle
        && !config.reject_loss_regression
}

fn mini_transformer_uses_train_core_step_for_model(
    config: MiniTransformerMlpTrainConfig,
    model: &MiniTransformerMlpModel,
) -> bool {
    model.transformer_layers() == 1 && mini_transformer_uses_train_core_step(config)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    InvalidConfig,
    InvalidModel(&'static str),
    CoreRejected(&'static str),
    TraceWrite,
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid training config"),
            Self::InvalidModel(message) => write!(f, "invalid model artifact: {message}"),
            Self::CoreRejected(stage) => write!(f, "nsrl-core rejected training stage: {stage}"),
            Self::TraceWrite => write!(f, "failed to write training trace"),
        }
    }
}

impl std::error::Error for TrainError {}

fn train_core_error_to_train_error(
    error: nsrl_train_core::TrainCoreError,
    stage: &'static str,
) -> TrainError {
    match error {
        nsrl_train_core::TrainCoreError::InvalidConfig
        | nsrl_train_core::TrainCoreError::InvalidShape => TrainError::InvalidConfig,
        nsrl_train_core::TrainCoreError::CoreRejected => TrainError::CoreRejected(stage),
    }
}

pub fn run_mini_transformer_mlp_swarm_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_swarm_training_from_model(tokens, config, swarm_config, model)
}

pub fn run_mini_transformer_mlp_swarm_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    run_mini_transformer_mlp_swarm_training_from_model_with_progress(
        tokens,
        config,
        swarm_config,
        base_model,
        0,
        |_| Ok(()),
    )
}

pub fn run_mini_transformer_mlp_swarm_training_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    mut progress: F,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
{
    if swarm_config.workers == 0 || config.stride == 0 || config.window_offset >= tokens.len() {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let worker_count = mini_transformer_swarm_effective_worker_count(
        config,
        swarm_config.workers,
        available_windows,
    );
    let mut worker_runs = mini_transformer_train_swarm_workers(
        tokens,
        config,
        swarm_config,
        &base_model,
        worker_count,
        progress_interval_batches,
        &mut progress,
    )?;
    if worker_runs.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    worker_runs.sort_by_key(|run| run.worker_index);

    let best_position = worker_runs
        .iter()
        .enumerate()
        .min_by_key(|(_, run)| {
            (
                run.run.trace.final_total_error,
                run.run.trace.final_probability_error_q15,
                run.run.trace.final_invalid_forward_count,
                run.worker_index,
            )
        })
        .map(|(position, _)| position)
        .ok_or(TrainError::InvalidConfig)?;
    let best_worker_index = worker_runs[best_position].worker_index;
    let model = worker_runs[best_position].run.model.clone();
    let swarm_model = MiniTransformerMlpSwarmModel::new(
        best_worker_index,
        worker_runs
            .iter()
            .map(|run| run.run.model.clone())
            .collect::<Vec<_>>(),
    )?;
    let final_model_hash = model.model_hash();
    let workers = worker_runs
        .iter()
        .map(|run| mini_transformer_swarm_worker_trace(run.worker_index, &run.run.trace))
        .collect::<Vec<_>>();

    let trace = MiniTransformerMlpSwarmTrainingTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: swarm_config.trace_detail,
        },
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        best_worker_index,
        final_model_hash,
        workers,
    };

    Ok(MiniTransformerMlpSwarmTrainingRun {
        trace,
        model,
        swarm_model,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_mini_transformer_mlp_swarm_worker_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    worker_index: usize,
    worker_count: usize,
    base_model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    mut progress: F,
) -> Result<MiniTransformerMlpSwarmWorkerTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
{
    if worker_count == 0
        || worker_index >= worker_count
        || config.stride == 0
        || config.window_offset >= tokens.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0
        || mini_transformer_swarm_effective_worker_count(config, worker_count, available_windows)
            != worker_count
    {
        return Err(TrainError::InvalidConfig);
    }

    let worker_config = mini_transformer_swarm_worker_config(config, worker_index, worker_count);
    let mut latest_progress = vec![None; worker_count];
    let mut worker_progress = |worker_progress: &MiniTransformerMlpTrainingProgressTrace| {
        latest_progress[worker_index] = Some(MiniTransformerMlpSwarmWorkerProgressTrace {
            worker_index,
            progress: worker_progress.clone(),
        });
        progress(&mini_transformer_swarm_training_progress_trace(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: worker_count,
                trace_detail,
            },
            worker_count,
            &latest_progress,
        ))
    };
    let base_model_hash = base_model.model_hash();
    let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
        tokens,
        worker_config,
        base_model,
        progress_interval_batches,
        trace_detail,
        &mut worker_progress,
    )?;
    let worker = mini_transformer_swarm_worker_trace(worker_index, &run.trace);
    let artifact = MiniTransformerMlpSwarmWorkerArtifact {
        worker_count,
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        base_max_windows: config.max_windows,
        base_model_hash,
        worker,
        model: run.model,
    };

    Ok(MiniTransformerMlpSwarmWorkerTrainingRun { artifact })
}

pub fn assemble_mini_transformer_mlp_swarm_worker_artifacts(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    base_model: &MiniTransformerMlpModel,
    artifacts: Vec<MiniTransformerMlpSwarmWorkerArtifact>,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    let first = artifacts.first().ok_or(TrainError::InvalidConfig)?;
    let worker_count = first.worker_count;
    if worker_count == 0
        || artifacts.len() != worker_count
        || config.stride == 0
        || config.window_offset >= tokens.len()
        || base_model.context_seq_len != config.seq_len
    {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0
        || mini_transformer_swarm_effective_worker_count(config, worker_count, available_windows)
            != worker_count
    {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let base_model_hash = base_model.model_hash();
    let mut slots = vec![None; worker_count];
    for artifact in artifacts {
        validate_mini_transformer_swarm_worker_artifact(
            tokens,
            config,
            base_model,
            worker_count,
            token_hash,
            base_model_hash,
            &artifact,
        )?;
        let worker_index = artifact.worker.worker_index;
        if slots[worker_index].is_some() {
            return Err(TrainError::InvalidConfig);
        }
        slots[worker_index] = Some(artifact);
    }

    let artifacts = slots
        .into_iter()
        .map(|slot| slot.ok_or(TrainError::InvalidConfig))
        .collect::<Result<Vec<_>, _>>()?;
    let best_position = artifacts
        .iter()
        .enumerate()
        .min_by_key(|(_, artifact)| {
            (
                artifact.worker.final_total_error,
                artifact.worker.final_probability_error_q15,
                artifact.worker.final_invalid_forward_count,
                artifact.worker.worker_index,
            )
        })
        .map(|(position, _)| position)
        .ok_or(TrainError::InvalidConfig)?;
    let best_worker_index = artifacts[best_position].worker.worker_index;
    let model = artifacts[best_position].model.clone();
    let swarm_model = MiniTransformerMlpSwarmModel::new(
        best_worker_index,
        artifacts
            .iter()
            .map(|artifact| artifact.model.clone())
            .collect(),
    )?;
    let workers = artifacts
        .iter()
        .map(|artifact| artifact.worker.clone())
        .collect::<Vec<_>>();
    let trace = MiniTransformerMlpSwarmTrainingTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: MiniTransformerTraceDetail::None,
        },
        token_count: tokens.len(),
        token_hash,
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        best_worker_index,
        final_model_hash: model.model_hash(),
        workers,
    };

    Ok(MiniTransformerMlpSwarmTrainingRun {
        trace,
        model,
        swarm_model,
    })
}

pub fn run_mini_transformer_mlp_swarm_scaling_benchmark(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    max_workers: usize,
    trace_detail: MiniTransformerTraceDetail,
) -> Result<MiniTransformerMlpSwarmScalingTrace, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_swarm_scaling_benchmark_from_model(
        tokens,
        config,
        max_workers,
        trace_detail,
        model,
    )
}

pub fn run_mini_transformer_mlp_swarm_scaling_benchmark_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    max_workers: usize,
    trace_detail: MiniTransformerTraceDetail,
    base_model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpSwarmScalingTrace, TrainError> {
    if max_workers == 0 {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let worker_counts = mini_transformer_swarm_scaling_worker_counts(max_workers);
    let available_parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut runs = Vec::with_capacity(worker_counts.len());
    let mut baseline_elapsed_ns = 0_u64;

    for &requested_worker_count in &worker_counts {
        let start = std::time::Instant::now();
        let run = run_mini_transformer_mlp_swarm_training_from_model(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: requested_worker_count,
                trace_detail,
            },
            base_model.clone(),
        )?;
        let elapsed_ns = mini_transformer_elapsed_ns_u64(start.elapsed());
        if baseline_elapsed_ns == 0 {
            baseline_elapsed_ns = elapsed_ns.max(1);
        }

        let examined_windows = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.examined_windows)
            .sum::<usize>();
        let updates = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.updates)
            .sum::<usize>();
        let accepted_batch_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.accepted_batch_count)
            .sum::<usize>();
        let rejected_batch_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.rejected_batch_count)
            .sum::<usize>();
        let rollback_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.rollback_count)
            .sum::<usize>();
        let best_worker = run
            .trace
            .workers
            .iter()
            .find(|worker| worker.worker_index == run.trace.best_worker_index)
            .ok_or(TrainError::InvalidConfig)?;
        let speedup_per_mille =
            mini_transformer_ratio_per_mille_u64(baseline_elapsed_ns, elapsed_ns.max(1));
        let parallel_efficiency_per_mille =
            speedup_per_mille / u64::try_from(run.trace.worker_count.max(1)).unwrap_or(u64::MAX);

        runs.push(MiniTransformerMlpSwarmScalingRunTrace {
            requested_worker_count,
            effective_worker_count: run.trace.worker_count,
            elapsed_ns,
            speedup_per_mille,
            parallel_efficiency_per_mille,
            windows_per_second_milli: mini_transformer_rate_per_second_milli(
                examined_windows,
                elapsed_ns,
            ),
            updates_per_second_milli: mini_transformer_rate_per_second_milli(updates, elapsed_ns),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            best_worker_index: run.trace.best_worker_index,
            best_final_total_error: best_worker.final_total_error,
            best_final_probability_error_q15: best_worker.final_probability_error_q15,
            best_final_accuracy_per_mille: best_worker.final_accuracy_per_mille,
            final_model_hash: run.trace.final_model_hash,
        });
    }

    Ok(MiniTransformerMlpSwarmScalingTrace {
        config,
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        available_parallelism,
        requested_max_workers: max_workers,
        worker_counts,
        runs,
    })
}

fn mini_transformer_swarm_scaling_worker_counts(max_workers: usize) -> Vec<usize> {
    let max_workers = max_workers.max(1);
    let mut counts = Vec::new();
    let mut worker_count = 1_usize;
    while worker_count < max_workers {
        counts.push(worker_count);
        worker_count = worker_count.saturating_mul(2);
        if worker_count == 0 {
            break;
        }
    }
    if counts.last().copied() != Some(max_workers) {
        counts.push(max_workers);
    }
    counts
}

fn mini_transformer_elapsed_ns_u64(elapsed: std::time::Duration) -> u64 {
    elapsed.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn mini_transformer_rate_per_second_milli(events: usize, elapsed_ns: u64) -> u64 {
    if elapsed_ns == 0 {
        return 0;
    }
    ((events as u128).saturating_mul(1_000_000_000_000_u128) / u128::from(elapsed_ns))
        .min(u128::from(u64::MAX)) as u64
}

fn mini_transformer_ratio_per_mille_u64(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    (u128::from(numerator).saturating_mul(1000) / u128::from(denominator)).min(u128::from(u64::MAX))
        as u64
}

struct MiniTransformerMlpSwarmWorkerRun {
    worker_index: usize,
    run: MiniTransformerMlpTrainingRun,
}

fn mini_transformer_train_swarm_workers(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: &MiniTransformerMlpModel,
    worker_count: usize,
    progress_interval_batches: usize,
    progress: &mut impl FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
) -> Result<Vec<MiniTransformerMlpSwarmWorkerRun>, TrainError> {
    if worker_count <= 1 {
        let worker_config = mini_transformer_swarm_worker_config(config, 0, 1);
        let mut latest_progress = vec![None; worker_count];
        let mut worker_progress = |worker_progress: &MiniTransformerMlpTrainingProgressTrace| {
            latest_progress[0] = Some(MiniTransformerMlpSwarmWorkerProgressTrace {
                worker_index: 0,
                progress: worker_progress.clone(),
            });
            progress(&mini_transformer_swarm_training_progress_trace(
                tokens,
                config,
                swarm_config,
                worker_count,
                &latest_progress,
            ))
        };
        let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
            tokens,
            worker_config,
            base_model.clone(),
            progress_interval_batches,
            swarm_config.trace_detail,
            &mut worker_progress,
        )?;
        return Ok(vec![MiniTransformerMlpSwarmWorkerRun {
            worker_index: 0,
            run,
        }]);
    }

    std::thread::scope(|scope| {
        let (progress_tx, progress_rx) =
            std::sync::mpsc::channel::<MiniTransformerMlpSwarmWorkerProgressTrace>();
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let worker_config =
                mini_transformer_swarm_worker_config(config, worker_index, worker_count);
            let worker_model = base_model.clone();
            let progress_tx = progress_tx.clone();
            handles.push(scope.spawn(move || {
                let mut worker_progress = |progress: &MiniTransformerMlpTrainingProgressTrace| {
                    progress_tx
                        .send(MiniTransformerMlpSwarmWorkerProgressTrace {
                            worker_index,
                            progress: progress.clone(),
                        })
                        .map_err(|_| TrainError::TraceWrite)
                };
                let run =
                    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
                        tokens,
                        worker_config,
                        worker_model,
                        progress_interval_batches,
                        swarm_config.trace_detail,
                        &mut worker_progress,
                    )?;
                Ok(MiniTransformerMlpSwarmWorkerRun { worker_index, run })
            }));
        }
        drop(progress_tx);

        let mut latest_progress = vec![None; worker_count];
        for worker_progress in progress_rx {
            let worker_index = worker_progress.worker_index;
            if worker_index < latest_progress.len() {
                latest_progress[worker_index] = Some(worker_progress);
                progress(&mini_transformer_swarm_training_progress_trace(
                    tokens,
                    config,
                    swarm_config,
                    worker_count,
                    &latest_progress,
                ))?;
            }
        }

        let mut runs = Vec::with_capacity(worker_count);
        for handle in handles {
            match handle.join() {
                Ok(result) => runs.push(result?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(runs)
    })
}

fn mini_transformer_swarm_training_progress_trace(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    worker_count: usize,
    latest_progress: &[Option<MiniTransformerMlpSwarmWorkerProgressTrace>],
) -> MiniTransformerMlpSwarmTrainingProgressTrace {
    MiniTransformerMlpSwarmTrainingProgressTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: swarm_config.trace_detail,
        },
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        workers: latest_progress
            .iter()
            .filter_map(|progress| progress.clone())
            .collect(),
    }
}

fn mini_transformer_swarm_effective_worker_count(
    config: MiniTransformerMlpTrainConfig,
    requested_workers: usize,
    available_windows: usize,
) -> usize {
    let by_requested = match config.max_windows {
        Some(max_windows) => requested_workers.min(max_windows.max(1)).max(1),
        None => requested_workers.max(1),
    };
    by_requested.min(available_windows.max(1)).max(1)
}

fn mini_transformer_swarm_worker_config(
    mut config: MiniTransformerMlpTrainConfig,
    worker_index: usize,
    worker_count: usize,
) -> MiniTransformerMlpTrainConfig {
    let base_stride = config.stride.max(1);
    config.window_offset = config
        .window_offset
        .saturating_add(worker_index.saturating_mul(base_stride));
    config.stride = base_stride.saturating_mul(worker_count.max(1));
    config.max_windows = config.max_windows.map(|max_windows| {
        mini_transformer_swarm_worker_window_limit(max_windows, worker_index, worker_count)
    });
    config
}

fn mini_transformer_swarm_worker_window_limit(
    max_windows: usize,
    worker_index: usize,
    worker_count: usize,
) -> usize {
    let base = max_windows / worker_count.max(1);
    let remainder = max_windows % worker_count.max(1);
    base + usize::from(worker_index < remainder)
}

fn mini_transformer_swarm_worker_trace(
    worker_index: usize,
    trace: &MiniTransformerMlpTrainingTrace,
) -> MiniTransformerMlpSwarmWorkerTrace {
    MiniTransformerMlpSwarmWorkerTrace {
        worker_index,
        window_offset: trace.config.window_offset,
        stride: trace.config.stride,
        max_windows: trace.config.max_windows,
        token_hash: trace.token_hash,
        window_hash: trace.window_hash,
        windows: trace.windows,
        examined_windows: trace.examined_windows,
        updates: trace.updates,
        accepted_batch_count: trace.accepted_batch_count,
        rejected_batch_count: trace.rejected_batch_count,
        rollback_count: trace.rollback_count,
        rejected_window_count: trace.rejected_window_count,
        final_invalid_forward_count: trace.final_invalid_forward_count,
        initial_total_error: trace.initial_total_error,
        final_total_error: trace.final_total_error,
        initial_probability_error_q15: trace.initial_probability_error_q15,
        final_probability_error_q15: trace.final_probability_error_q15,
        final_accuracy_per_mille: trace.final_accuracy_per_mille,
        final_model_hash: trace.final_model_hash,
        final_logits_hash: trace.final_logits_hash,
    }
}

fn validate_mini_transformer_swarm_worker_artifact(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    base_model: &MiniTransformerMlpModel,
    worker_count: usize,
    token_hash: u64,
    base_model_hash: u64,
    artifact: &MiniTransformerMlpSwarmWorkerArtifact,
) -> Result<(), TrainError> {
    let worker_index = artifact.worker.worker_index;
    if artifact.worker_count != worker_count
        || artifact.token_count != tokens.len()
        || artifact.token_hash != token_hash
        || artifact.base_window_offset != config.window_offset
        || artifact.base_stride != config.stride
        || artifact.base_max_windows != config.max_windows
        || artifact.base_model_hash != base_model_hash
        || worker_index >= worker_count
        || artifact.model.context_seq_len != config.seq_len
        || artifact.model.model_hash() != artifact.worker.final_model_hash
    {
        return Err(TrainError::InvalidConfig);
    }

    let worker_config = mini_transformer_swarm_worker_config(config, worker_index, worker_count);
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, worker_config);
    if starts.is_empty()
        || artifact.worker.window_offset != worker_config.window_offset
        || artifact.worker.stride != worker_config.stride
        || artifact.worker.max_windows != worker_config.max_windows
        || artifact.worker.token_hash != token_hash
        || artifact.worker.window_hash
            != hash_mini_transformer_windows(tokens, worker_config, &starts)
        || artifact.worker.windows != starts.len()
    {
        return Err(TrainError::InvalidConfig);
    }

    let initial_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        base_model,
        worker_config.seq_len,
        worker_config.attention_kind,
        worker_config.position_policy,
    )?;
    let final_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        &artifact.model,
        worker_config.seq_len,
        worker_config.attention_kind,
        worker_config.position_policy,
    )?;
    let final_accuracy_per_mille =
        starts.len().saturating_sub(final_eval.mistakes) * 1000 / starts.len();
    if artifact.worker.initial_total_error != initial_eval.mistakes
        || artifact.worker.initial_probability_error_q15 != initial_eval.probability_error_q15
        || artifact.worker.final_total_error != final_eval.mistakes
        || artifact.worker.final_probability_error_q15 != final_eval.probability_error_q15
        || artifact.worker.final_invalid_forward_count != final_eval.invalid_forward_count
        || artifact.worker.final_logits_hash != final_eval.logits_hash
        || artifact.worker.final_accuracy_per_mille != final_accuracy_per_mille
    {
        return Err(TrainError::InvalidConfig);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct MiniTransformerAdamStateRanges {
    embeddings: Range<usize>,
    position_embeddings: Range<usize>,
    attention_rms: Range<usize>,
    mlp_rms: Range<usize>,
    q: Range<usize>,
    k: Range<usize>,
    v: Range<usize>,
    o: Range<usize>,
    up: Range<usize>,
    gate: Range<usize>,
    down: Range<usize>,
    output: Range<usize>,
}

#[derive(Debug, Clone)]
struct MiniTransformerAdamBatchUpdateStats {
    output_head: LinearWeightUpdateStats,
    mlp: GatedMlpWeightUpdateStats,
    embedding: SoftmaxUpdateStats,
    rms_norm: SoftmaxUpdateStats,
    attention: MiniTransformerAttentionWeightUpdateStats,
}

fn mini_transformer_adam_state_ranges(
    model: &MiniTransformerMlpModel,
) -> Result<MiniTransformerAdamStateRanges, TrainError> {
    fn take(cursor: &mut usize, len: usize) -> Result<Range<usize>, TrainError> {
        let start = *cursor;
        let end = start
            .checked_add(len)
            .ok_or(TrainError::InvalidModel("optimizer range overflow"))?;
        *cursor = end;
        Ok(start..end)
    }

    let mut cursor = 0_usize;
    let ranges = MiniTransformerAdamStateRanges {
        embeddings: take(&mut cursor, model.embeddings.len())?,
        position_embeddings: take(&mut cursor, model.position_embeddings.len())?,
        attention_rms: take(&mut cursor, model.attention_rms_weights.len())?,
        mlp_rms: take(&mut cursor, model.mlp_rms_weights.len())?,
        q: take(&mut cursor, model.q_weights.len())?,
        k: take(&mut cursor, model.k_weights.len())?,
        v: take(&mut cursor, model.v_weights.len())?,
        o: take(&mut cursor, model.o_weights.len())?,
        up: take(&mut cursor, model.up_weights.len())?,
        gate: take(&mut cursor, model.gate_weights.len())?,
        down: take(&mut cursor, model.down_weights.len())?,
        output: take(&mut cursor, model.output_weights.len())?,
    };
    if cursor != model.optimizer_parameter_count()? {
        return Err(TrainError::InvalidModel("optimizer range mismatch"));
    }
    Ok(ranges)
}

fn offset_optimizer_range(
    tensor: &Range<usize>,
    local: &Range<usize>,
) -> Result<Range<usize>, TrainError> {
    if local.start > local.end || local.end > tensor.len() {
        return Err(TrainError::InvalidConfig);
    }
    let start = tensor
        .start
        .checked_add(local.start)
        .ok_or(TrainError::InvalidConfig)?;
    let end = tensor
        .start
        .checked_add(local.end)
        .ok_or(TrainError::InvalidConfig)?;
    Ok(start..end)
}

fn apply_integer_adam_state_slice_i8(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i8],
    state: &mut MiniTransformerAdamOptimizerState,
    state_range: Range<usize>,
) -> Result<LinearWeightUpdateStats, TrainError> {
    if accumulators.len() != weights.len() || state_range.len() != weights.len() {
        return Err(TrainError::InvalidConfig);
    }
    let config = state.config;
    let mut workspace = nsrl_train_core::IntegerAdamStateWorkspace {
        step: state.step,
        first_moments: &mut state.first_moments[state_range.clone()],
        second_moments: &mut state.second_moments[state_range.clone()],
        update_residuals: &mut state.update_residuals[state_range],
    };
    nsrl_train_core::apply_integer_adam_accumulators_i64_to_i8(
        accumulators,
        sample_count,
        weights,
        config,
        &mut workspace,
    )
    .map_err(|error| train_core_error_to_train_error(error, "integer_adam_i8_apply"))
}

fn apply_integer_adam_state_slice_i16(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i16],
    state: &mut MiniTransformerAdamOptimizerState,
    state_range: Range<usize>,
) -> Result<LinearWeightUpdateStats, TrainError> {
    if accumulators.len() != weights.len() || state_range.len() != weights.len() {
        return Err(TrainError::InvalidConfig);
    }
    let config = state.config;
    let mut workspace = nsrl_train_core::IntegerAdamStateWorkspace {
        step: state.step,
        first_moments: &mut state.first_moments[state_range.clone()],
        second_moments: &mut state.second_moments[state_range.clone()],
        update_residuals: &mut state.update_residuals[state_range],
    };
    nsrl_train_core::apply_integer_adam_accumulators_i64_to_i16(
        accumulators,
        sample_count,
        weights,
        config,
        &mut workspace,
    )
    .map_err(|error| train_core_error_to_train_error(error, "integer_adam_i16_apply"))
}

fn mini_transformer_apply_integer_adam_batch(
    batch: &MiniTransformerMapReduceBatchResult,
    model: &mut MiniTransformerMlpModel,
    state: &mut MiniTransformerAdamOptimizerState,
    position_policy: MiniTransformerPositionPolicy,
    train_scope: MiniTransformerAdamTrainScope,
) -> Result<MiniTransformerAdamBatchUpdateStats, TrainError> {
    state.validate_for_model(model)?;
    let ranges = mini_transformer_adam_state_ranges(model)?;
    let output_head = if matches!(
        train_scope,
        MiniTransformerAdamTrainScope::All
            | MiniTransformerAdamTrainScope::Output
            | MiniTransformerAdamTrainScope::FinalMlpAndOutput
    ) {
        apply_integer_adam_state_slice_i8(
            &batch.output_head_gradient.accumulators,
            batch.output_head_gradient.sample_count,
            &mut model.output_weights,
            state,
            ranges.output.clone(),
        )?
    } else {
        empty_linear_weight_update_stats()
    };
    let mut mlp = empty_gated_mlp_weight_update_stats();
    let mut attention = empty_mini_transformer_attention_weight_update_stats();
    let mut rms_linear = empty_linear_weight_update_stats();
    let layers = model.checked_transformer_layers()?;
    if batch.mlp_weight_gradients.len() != layers
        || batch.attention_weight_gradients.len() != layers
        || batch.rms_weight_gradients.len() != layers
    {
        return Err(TrainError::InvalidConfig);
    }
    for layer_index in 0..layers {
        let attention_local = model.attention_weight_range(layer_index)?;
        let q_state_range = offset_optimizer_range(&ranges.q, &attention_local)?;
        let k_state_range = offset_optimizer_range(&ranges.k, &attention_local)?;
        let v_state_range = offset_optimizer_range(&ranges.v, &attention_local)?;
        let o_state_range = offset_optimizer_range(&ranges.o, &attention_local)?;
        let up_local = model.mlp_up_or_gate_weight_range(layer_index)?;
        let down_local = model.mlp_down_weight_range(layer_index)?;
        if train_scope == MiniTransformerAdamTrainScope::All {
            let attention_gradient = &batch.attention_weight_gradients[layer_index];
            let q = apply_integer_adam_state_slice_i8(
                &attention_gradient.q.accumulators,
                attention_gradient.q.sample_count,
                &mut model.q_weights[attention_local.clone()],
                state,
                q_state_range,
            )?;
            let k = apply_integer_adam_state_slice_i8(
                &attention_gradient.k.accumulators,
                attention_gradient.k.sample_count,
                &mut model.k_weights[attention_local.clone()],
                state,
                k_state_range,
            )?;
            let v = apply_integer_adam_state_slice_i8(
                &attention_gradient.v.accumulators,
                attention_gradient.v.sample_count,
                &mut model.v_weights[attention_local.clone()],
                state,
                v_state_range,
            )?;
            let o = apply_integer_adam_state_slice_i8(
                &attention_gradient.o.accumulators,
                attention_gradient.o.sample_count,
                &mut model.o_weights[attention_local.clone()],
                state,
                o_state_range,
            )?;
            let mut projection_total = empty_linear_weight_update_stats();
            for stats in [q, k, v, o] {
                add_linear_weight_update_stats_checked(&mut projection_total, stats)?;
            }
            add_mini_transformer_attention_weight_update_stats_checked(
                &mut attention,
                MiniTransformerAttentionWeightUpdateStats {
                    q,
                    k,
                    v,
                    o,
                    gradient_saturation_count: projection_total.gradient_saturation_count,
                    zero_delta_count: projection_total.zero_delta_count,
                    weight_delta_l1: projection_total.weight_delta_l1,
                    grad_embedding_output: Vec::new(),
                },
            )?;
        }

        if train_scope == MiniTransformerAdamTrainScope::All
            || (matches!(
                train_scope,
                MiniTransformerAdamTrainScope::FinalMlp
                    | MiniTransformerAdamTrainScope::FinalMlpAndOutput
            ) && layer_index + 1 == layers)
        {
            let mlp_gradient = &batch.mlp_weight_gradients[layer_index];
            let layer_mlp = GatedMlpWeightUpdateStats {
                down: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.down.accumulators,
                    mlp_gradient.down.sample_count,
                    &mut model.down_weights[down_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.down, &down_local)?,
                )?,
                up: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.up.accumulators,
                    mlp_gradient.up.sample_count,
                    &mut model.up_weights[up_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.up, &up_local)?,
                )?,
                gate: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.gate.accumulators,
                    mlp_gradient.gate.sample_count,
                    &mut model.gate_weights[up_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.gate, &up_local)?,
                )?,
            };
            add_gated_mlp_weight_update_stats_checked(&mut mlp, layer_mlp)?;
        }
        if matches!(
            train_scope,
            MiniTransformerAdamTrainScope::All | MiniTransformerAdamTrainScope::RmsNorm
        ) && model.rms_norm_enabled()
        {
            let rms_local = model.rms_weight_range(layer_index)?;
            let rms_gradient = &batch.rms_weight_gradients[layer_index];
            let attention_rms = apply_integer_adam_state_slice_i16(
                &rms_gradient.attention.accumulators,
                rms_gradient.attention.sample_count,
                &mut model.attention_rms_weights[rms_local.clone()],
                state,
                offset_optimizer_range(&ranges.attention_rms, &rms_local)?,
            )?;
            let mlp_rms = apply_integer_adam_state_slice_i16(
                &rms_gradient.mlp.accumulators,
                rms_gradient.mlp.sample_count,
                &mut model.mlp_rms_weights[rms_local.clone()],
                state,
                offset_optimizer_range(&ranges.mlp_rms, &rms_local)?,
            )?;
            add_linear_weight_update_stats_checked(&mut rms_linear, attention_rms)?;
            add_linear_weight_update_stats_checked(&mut rms_linear, mlp_rms)?;
        }
    }

    let mut embedding_linear = empty_linear_weight_update_stats();
    if train_scope == MiniTransformerAdamTrainScope::All {
        embedding_linear = apply_integer_adam_state_slice_i16(
            &batch.embedding_gradient.token_accumulators,
            batch.embedding_gradient.sample_count,
            &mut model.embeddings,
            state,
            ranges.embeddings,
        )?;
        if position_policy.uses_position_embeddings() {
            let position = apply_integer_adam_state_slice_i16(
                &batch.embedding_gradient.position_accumulators,
                batch.embedding_gradient.sample_count,
                &mut model.position_embeddings,
                state,
                ranges.position_embeddings,
            )?;
            add_linear_weight_update_stats_checked(&mut embedding_linear, position)?;
        }
    }
    let embedding = SoftmaxUpdateStats {
        gradient_saturation_count: embedding_linear.gradient_saturation_count,
        zero_delta_count: embedding_linear.zero_delta_count,
        weight_delta_l1: embedding_linear.weight_delta_l1,
    };
    let rms_norm = SoftmaxUpdateStats {
        gradient_saturation_count: rms_linear.gradient_saturation_count,
        zero_delta_count: rms_linear.zero_delta_count,
        weight_delta_l1: rms_linear.weight_delta_l1,
    };
    // All tensor slices represent one optimizer batch; slice-local workspace
    // steps are intentionally ignored and the global step advances exactly once.
    state.step = state
        .step
        .checked_add(1)
        .ok_or(TrainError::CoreRejected("integer_adam_step_overflow"))?;
    state.bind_to_model(model)?;
    Ok(MiniTransformerAdamBatchUpdateStats {
        output_head,
        mlp,
        embedding,
        rms_norm,
        attention,
    })
}

pub fn run_mini_transformer_mlp_integer_adam_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens,
        config,
        optimizer_config,
        model,
        None,
    )
}

pub fn run_mini_transformer_mlp_integer_adam_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
    model: MiniTransformerMlpModel,
    optimizer_state: Option<MiniTransformerAdamOptimizerState>,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer_config,
        model,
        optimizer_state,
        MiniTransformerAdamTrainScope::All,
    )
}

pub fn run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
    mut model: MiniTransformerMlpModel,
    optimizer_state: Option<MiniTransformerAdamOptimizerState>,
    train_scope: MiniTransformerAdamTrainScope,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.batch_windows == 0
        || config.target_token_min > config.target_token_max
        || !valid_mini_transformer_target_segment(config.target_segment)
        || !valid_q15_weight_floor(config.target_frequency_min_weight_q15)
        || config.argmax_margin_weight_q15 < 0
        || !optimizer_config.is_valid()
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_q_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind.uses_incremental_state()
        || config.attention_vo_error_feedback
        || config.attention_vo_oracle
        || config.adaptive_shift_controller_enabled()
        || model.context_seq_len != config.seq_len
    {
        return Err(TrainError::InvalidConfig);
    }
    model.checked_transformer_layers()?;
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
        tokens,
        &starts,
        config.seq_len,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )?;
    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let initial_model_hash = model.model_hash();
    let initial_mistakes = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let initial_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let mut optimizer_state = match optimizer_state {
        Some(state) => {
            state.validate_for_model(&model)?;
            if state.config != optimizer_config {
                return Err(TrainError::InvalidConfig);
            }
            state
        }
        None => MiniTransformerAdamOptimizerState::new_for_model(&model, optimizer_config)?,
    };
    #[cfg(feature = "mini-calibrated")]
    if config.position_policy == MiniTransformerPositionPolicy::Nope
        && mini_transformer_suffix_memory_is_installed(&model.position_embeddings)
    {
        for weight in model
            .position_embeddings
            .iter_mut()
            .take(MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC.len() / 2)
        {
            *weight = 0;
        }
        optimizer_state.bind_to_model(&model)?;
    }
    let mut examined_windows = 0_usize;
    let mut updates = 0_usize;
    let mut accepted_batch_count = 0_usize;
    let mut rejected_batch_count = 0_usize;
    let mut output_head_delta_l1 = 0_u64;
    let mut mlp_delta_l1 = 0_u64;
    let mut embedding_delta_l1 = 0_u64;
    let mut rms_norm_delta_l1 = 0_u64;
    let mut attention_delta_l1 = 0_u64;
    let mut attention_q_delta_l1 = 0_u64;
    let mut attention_k_delta_l1 = 0_u64;
    let mut attention_v_delta_l1 = 0_u64;
    let mut attention_o_delta_l1 = 0_u64;
    let mut mlp_saturation_count = 0_usize;
    let mut attention_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;

    for epoch in 0..config.epochs {
        let mut batch_start = 0_usize;
        while batch_start < starts.len() {
            let batch_end = batch_start
                .saturating_add(config.batch_windows)
                .min(starts.len());
            examined_windows = examined_windows.saturating_add(batch_end - batch_start);
            let batch_result = if config.batch_mode == MiniTransformerBatchMode::MapReduce {
                mini_transformer_map_reduce_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start,
                    batch_end,
                    epoch,
                    &model,
                    config,
                    updates,
                    MiniTransformerTraceDetail::None,
                    usize::MAX,
                )
            } else {
                mini_transformer_map_reduce_worker_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start,
                    batch_end,
                    batch_start,
                    epoch,
                    &model,
                    config,
                    updates,
                    MiniTransformerTraceDetail::None,
                    usize::MAX,
                )
            };
            let batch_result = match batch_result {
                Ok(result) if result.accepted_window_count > 0 => result,
                Ok(_) | Err(TrainError::CoreRejected(_)) => {
                    rejected_batch_count = rejected_batch_count.saturating_add(1);
                    batch_start = batch_end;
                    continue;
                }
                Err(error) => return Err(error),
            };
            mlp_saturation_count =
                mlp_saturation_count.saturating_add(batch_result.mlp_saturation_count);
            attention_saturation_count =
                attention_saturation_count.saturating_add(batch_result.attention_saturation_count);
            residual_saturation_count =
                residual_saturation_count.saturating_add(batch_result.residual_saturation_count);

            let mut candidate_model = model.clone();
            let mut candidate_state = optimizer_state.clone();
            let update = mini_transformer_apply_integer_adam_batch(
                &batch_result,
                &mut candidate_model,
                &mut candidate_state,
                config.position_policy,
                train_scope,
            )?;
            let batch_starts = &starts[batch_start..batch_end];
            let batch_valid = mini_transformer_validate_batch_windows(
                &candidate_model,
                tokens,
                batch_starts,
                config.seq_len,
                config.attention_kind,
                config.position_policy,
            )
            .and_then(|_| {
                mini_transformer_validate_guard_windows(
                    &candidate_model,
                    tokens,
                    &starts,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                    epoch,
                    batch_end - 1,
                    config.epochs,
                )
            })
            .is_ok();
            let loss_regressed = if batch_valid && config.reject_loss_regression {
                let guard = mini_transformer_loss_guard_starts(&starts, batch_start, batch_end);
                let before = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                    tokens,
                    &guard,
                    &model,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                )?;
                match mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                    tokens,
                    &guard,
                    &candidate_model,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                ) {
                    Ok(after) => mini_transformer_loss_guard_regressed(before, after, guard.len()),
                    Err(TrainError::CoreRejected(_)) => true,
                    Err(error) => return Err(error),
                }
            } else {
                false
            };
            if batch_valid && !loss_regressed {
                model = candidate_model;
                optimizer_state = candidate_state;
                updates = updates.saturating_add(batch_result.accepted_window_count);
                accepted_batch_count = accepted_batch_count.saturating_add(1);
                output_head_delta_l1 =
                    output_head_delta_l1.saturating_add(update.output_head.weight_delta_l1);
                mlp_delta_l1 =
                    mlp_delta_l1.saturating_add(update.mlp.weight_delta_l1().unwrap_or(0));
                embedding_delta_l1 =
                    embedding_delta_l1.saturating_add(update.embedding.weight_delta_l1);
                rms_norm_delta_l1 =
                    rms_norm_delta_l1.saturating_add(update.rms_norm.weight_delta_l1);
                attention_delta_l1 =
                    attention_delta_l1.saturating_add(update.attention.weight_delta_l1);
                attention_q_delta_l1 =
                    attention_q_delta_l1.saturating_add(update.attention.q.weight_delta_l1);
                attention_k_delta_l1 =
                    attention_k_delta_l1.saturating_add(update.attention.k.weight_delta_l1);
                attention_v_delta_l1 =
                    attention_v_delta_l1.saturating_add(update.attention.v.weight_delta_l1);
                attention_o_delta_l1 =
                    attention_o_delta_l1.saturating_add(update.attention.o.weight_delta_l1);
            } else {
                rejected_batch_count = rejected_batch_count.saturating_add(1);
            }
            batch_start = batch_end;
        }
    }

    #[cfg(feature = "mini-calibrated")]
    if config.position_policy == MiniTransformerPositionPolicy::Nope
        && train_scope == MiniTransformerAdamTrainScope::All
    {
        mini_transformer_install_ngram_cache(&mut model, tokens)?;
    }
    optimizer_state.bind_to_model(&model)?;
    let final_mistakes = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let final_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let optimizer_state_hash = optimizer_state.state_hash()?;
    Ok(MiniTransformerAdamTrainingRun {
        trace: MiniTransformerAdamTrainingTrace {
            schema: MINI_TRANSFORMER_ADAM_SCHEMA,
            config,
            optimizer_config,
            train_scope,
            token_count: tokens.len(),
            token_hash,
            window_hash,
            windows: starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            initial_mistakes,
            final_mistakes,
            initial_probability_error_q15,
            final_probability_error_q15,
            transformer_layers: model.transformer_layers(),
            rms_norm_enabled: model.rms_norm_enabled(),
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            rms_norm_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            mlp_saturation_count,
            attention_saturation_count,
            residual_saturation_count,
            initial_model_hash,
            final_model_hash: model.model_hash(),
            optimizer_step: optimizer_state.step,
            optimizer_state_hash,
        },
        model,
        optimizer_state,
    })
}

pub fn run_mini_transformer_mlp_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingTrace, TrainError> {
    Ok(run_mini_transformer_mlp_training_with_model(tokens, config)?.trace)
}

pub const MINI_TRANSFORMER_EVAL_SCHEMA: &str = "nsrl.mini_transformer_eval.v1";
pub const MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerMlpEvalConfig {
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub attention_kind: MiniTransformerAttentionKind,
    pub position_policy: MiniTransformerPositionPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpEvalTrace {
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub config: MiniTransformerMlpEvalConfig,
    pub model_hash: u64,
    pub mistakes: usize,
    pub accuracy_per_mille: usize,
    pub probability_error_q15: usize,
    pub mean_probability_error_q15: usize,
    pub invalid_forward_count: usize,
    pub unique_predicted_tokens: usize,
    pub most_predicted_token: Option<u8>,
    pub most_predicted_token_count: usize,
    pub most_predicted_token_share_per_mille: usize,
    pub logits_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpWindowEvalRecord {
    pub start: usize,
    pub end: usize,
    pub mistakes: usize,
    pub probability_error_q15: usize,
    pub invalid_forward_count: usize,
    pub predicted_token: Option<u8>,
    pub last_hidden_q15: [i16; MINI_TRANSFORMER_D_MODEL],
    pub router_hidden_features_q15: [i16; MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES],
    pub logits_q8: Option<[i32; BYTE_VOCAB]>,
}

impl MiniTransformerMlpEvalTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_EVAL_SCHEMA);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        comma(&mut out);
        push_hash_field(&mut out, "window_hash", self.window_hash);
        comma(&mut out);
        push_usize_field(&mut out, "windows", self.windows);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_hash_field(&mut out, "hash", self.model_hash);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"evaluation\":{");
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "mistakes", self.mistakes);
        comma(&mut out);
        push_usize_field(&mut out, "accuracy_per_mille", self.accuracy_per_mille);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "probability_error_q15",
            self.probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mean_probability_error_q15",
            self.mean_probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "invalid_forward_count",
            self.invalid_forward_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "unique_predicted_tokens",
            self.unique_predicted_tokens,
        );
        comma(&mut out);
        push_optional_usize_field(
            &mut out,
            "most_predicted_token",
            self.most_predicted_token.map(usize::from),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "most_predicted_token_count",
            self.most_predicted_token_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "most_predicted_token_share_per_mille",
            self.most_predicted_token_share_per_mille,
        );
        comma(&mut out);
        push_hash_field(&mut out, "logits_hash", self.logits_hash);
        out.push_str("}}\n");
        out
    }
}

pub fn evaluate_mini_transformer_mlp_model(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpEvalConfig,
) -> Result<MiniTransformerMlpEvalTrace, TrainError> {
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || matches!(
            config.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingNope
                | MiniTransformerAttentionKind::LinearStreamingTttNope
        )
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts_config = MiniTransformerMlpTrainConfig {
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        attention_kind: config.attention_kind,
        position_policy: config.position_policy,
        ..MiniTransformerMlpTrainConfig::default()
    };
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, starts_config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let summary = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let windows = starts.len();
    Ok(MiniTransformerMlpEvalTrace {
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        window_hash: hash_mini_transformer_windows(tokens, starts_config, &starts),
        windows,
        config,
        model_hash: model.model_hash(),
        mistakes: summary.mistakes,
        accuracy_per_mille: windows.saturating_sub(summary.mistakes) * 1000 / windows,
        probability_error_q15: summary.probability_error_q15,
        mean_probability_error_q15: summary.probability_error_q15 / windows,
        invalid_forward_count: summary.invalid_forward_count,
        unique_predicted_tokens: summary.unique_predicted_tokens,
        most_predicted_token: summary.most_predicted_token,
        most_predicted_token_count: summary.most_predicted_token_count,
        most_predicted_token_share_per_mille: summary
            .most_predicted_token_count
            .saturating_mul(1000)
            / windows,
        logits_hash: summary.logits_hash,
    })
}

pub fn evaluate_mini_transformer_mlp_windows(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpEvalConfig,
) -> Result<Vec<MiniTransformerMlpWindowEvalRecord>, TrainError> {
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || matches!(
            config.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingNope
                | MiniTransformerAttentionKind::LinearStreamingTttNope
        )
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts_config = MiniTransformerMlpTrainConfig {
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        attention_kind: config.attention_kind,
        position_policy: config.position_policy,
        ..MiniTransformerMlpTrainConfig::default()
    };
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, starts_config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    mini_transformer_window_eval_records_with_attention_and_position_policy(
        tokens,
        &starts,
        model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )
}

pub fn run_mini_transformer_mlp_training_with_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_training_from_model(tokens, config, model)
}

pub fn run_mini_transformer_mlp_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    run_mini_transformer_mlp_training_from_model_with_progress(tokens, config, model, 0, |_| Ok(()))
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    progress: F,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
{
    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
        tokens,
        config,
        model,
        progress_interval_batches,
        MiniTransformerTraceDetail::Full,
        progress,
    )
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    progress: F,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
{
    run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace(
        tokens,
        config,
        model,
        progress_interval_batches,
        trace_detail,
        progress,
        |_| Ok(()),
    )
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace<
    F,
    G,
>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    mut model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    mut progress: F,
    mut binary_trace: G,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
    G: FnMut(MiniTransformerBinaryTraceRecord<'_>) -> Result<(), TrainError>,
{
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.batch_windows == 0
        || config.learning_rate <= 0
        || config.target_token_min > config.target_token_max
        || !valid_mini_transformer_target_segment(config.target_segment)
        || !valid_q15_weight_floor(config.target_frequency_min_weight_q15)
        || config.argmax_margin_weight_q15 < 0
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_q_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
        || config.attention_kind == MiniTransformerAttentionKind::LinearStreamingTttNope
        || (config.attention_vo_oracle && config.batch_windows <= 1)
        || (config.attention_vo_oracle
            && MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL)
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_mini_transformer_batch_mode(config)?;
    validate_mini_transformer_effective_learning_rate_shifts(config)?;
    if model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }
    if model.rms_norm_enabled() {
        return Err(TrainError::InvalidConfig);
    }
    let transformer_layers = model.checked_transformer_layers()?;
    let use_stacked_serial_backprop =
        transformer_layers > 1 && config.batch_mode == MiniTransformerBatchMode::Serial;

    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
        tokens,
        &starts,
        config.seq_len,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )?;
    let initial_model_hash = model.model_hash();
    let initial_embedding_hash = model.embedding_hash();
    let initial_output_head_hash = model.output_head_hash();
    let initial_mlp_hash = model.mlp_hash();
    let initial_attention_hash = model.attention_hash();
    let initial_attention_q_hash = model.attention_q_hash();
    let initial_attention_k_hash = model.attention_k_hash();
    let initial_attention_v_hash = model.attention_v_hash();
    let initial_attention_o_hash = model.attention_o_hash();
    let initial_total_error = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let initial_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let initial_mistakes = initial_total_error;
    binary_trace(MiniTransformerBinaryTraceRecord::Header { initial_model_hash })?;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut accepted_batch_count = 0_usize;
    let mut rejected_batch_count = 0_usize;
    let mut output_head_accumulator_batch_count = 0_usize;
    let mut output_head_accumulator_window_count = 0_usize;
    let mut mlp_accumulator_batch_count = 0_usize;
    let mut mlp_accumulator_window_count = 0_usize;
    let mut attention_accumulator_batch_count = 0_usize;
    let mut attention_accumulator_window_count = 0_usize;
    let mut embedding_accumulator_batch_count = 0_usize;
    let mut embedding_accumulator_window_count = 0_usize;
    let mut rollback_count = 0_usize;
    let mut rejected_window_count = 0_usize;
    let mut loss_regression_rejected_batch_count = 0_usize;
    let mut output_head_saturation_count = 0_usize;
    let mut mlp_saturation_count = 0_usize;
    let mut embedding_saturation_count = 0_usize;
    let mut attention_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;
    let mut output_head_zero_delta_count = 0_usize;
    let mut mlp_zero_delta_count = 0_usize;
    let mut embedding_zero_delta_count = 0_usize;
    let mut attention_zero_delta_count = 0_usize;
    let mut output_head_delta_l1 = 0_u64;
    let mut mlp_delta_l1 = 0_u64;
    let mut embedding_delta_l1 = 0_u64;
    let mut attention_delta_l1 = 0_u64;
    let mut attention_q_delta_l1 = 0_u64;
    let mut attention_k_delta_l1 = 0_u64;
    let mut attention_v_delta_l1 = 0_u64;
    let mut attention_o_delta_l1 = 0_u64;
    let mut output_head_carry_l1 = 0_u64;
    let mut mlp_carry_l1 = 0_u64;
    let mut embedding_carry_l1 = 0_u64;
    let mut attention_carry_l1 = 0_u64;
    let mut attention_q_carry_l1 = 0_u64;
    let mut attention_k_carry_l1 = 0_u64;
    let mut attention_v_carry_l1 = 0_u64;
    let mut attention_o_carry_l1 = 0_u64;
    let mut steps = Vec::new();
    let trace_sample_interval =
        mini_transformer_trace_sample_interval(progress_interval_batches, config.batch_windows);
    let mut rollback_history = vec![model.clone()];
    let mut output_head_gradient =
        LinearWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, BYTE_VOCAB)
            .ok_or(TrainError::InvalidConfig)?;
    let mut mlp_weight_gradients =
        mini_transformer_new_gated_mlp_weight_gradients(transformer_layers)?;
    let mut attention_weight_gradients =
        mini_transformer_new_attention_weight_gradients(transformer_layers)?;
    let mut embedding_gradient = MiniTransformerEmbeddingGradientI64::new(config.seq_len)
        .ok_or(TrainError::InvalidConfig)?;
    let mut adaptive_attention_shifts = MiniTransformerAdaptiveShiftState::new(config);
    let mut adaptive_shift_events = Vec::new();
    let adaptive_shift_controller_enabled = config.adaptive_shift_controller_enabled();
    let use_output_head_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_mlp_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_attention_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_embedding_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_train_core_step = mini_transformer_uses_train_core_step_for_model(config, &model);
    let mut train_core_workspace = if use_train_core_step {
        Some(MiniTransformerHostTrainCoreWorkspaceBuffers::new(
            config.seq_len,
        )?)
    } else {
        None
    };
    let mut host_training_workspace = if use_train_core_step {
        None
    } else {
        Some(MiniTransformerHostTrainCoreWorkspaceBuffers::new(
            config.seq_len,
        )?)
    };
    if progress_interval_batches > 0 {
        progress(&mini_transformer_training_progress_trace(
            config,
            tokens.len(),
            token_hash,
            window_hash,
            starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            rejected_window_count,
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            output_head_carry_l1,
            mlp_carry_l1,
            embedding_carry_l1,
            attention_carry_l1,
            attention_q_carry_l1,
            attention_k_carry_l1,
            attention_v_carry_l1,
            attention_o_carry_l1,
            &adaptive_attention_shifts,
            &model,
        ))?;
    }

    for epoch in 0..config.epochs {
        let mut batch_start_index = 0_usize;
        while batch_start_index < starts.len() {
            let batch_end_index = batch_start_index
                .saturating_add(config.batch_windows)
                .min(starts.len());
            let batch_model_checkpoint = model.clone();
            let updates_before_batch = updates;
            let steps_before_batch = steps.len();
            let rollbacks_before_batch = rollback_count;

            if config.batch_mode == MiniTransformerBatchMode::MapReduce {
                let batch_window_count = batch_end_index.saturating_sub(batch_start_index);
                examined_windows = examined_windows.saturating_add(batch_window_count);
                output_head_gradient.clear();
                mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                    &mut mlp_weight_gradients,
                );
                mini_transformer_clear_attention_weight_gradient_i64_layers(
                    &mut attention_weight_gradients,
                );
                embedding_gradient.clear();

                match mini_transformer_map_reduce_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start_index,
                    batch_end_index,
                    epoch,
                    &model,
                    config,
                    updates_before_batch,
                    trace_detail,
                    trace_sample_interval,
                ) {
                    Ok(batch_result) => {
                        mini_transformer_merge_linear_weight_gradient_i64(
                            &mut output_head_gradient,
                            &batch_result.output_head_gradient,
                        )?;
                        mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
                            &mut mlp_weight_gradients,
                            &batch_result.mlp_weight_gradients,
                        )?;
                        mini_transformer_merge_attention_weight_gradient_i64_layers(
                            &mut attention_weight_gradients,
                            &batch_result.attention_weight_gradients,
                        )?;
                        mini_transformer_merge_embedding_gradient_i64(
                            &mut embedding_gradient,
                            &batch_result.embedding_gradient,
                        )?;
                        updates = updates.saturating_add(batch_result.accepted_window_count);
                        mlp_saturation_count =
                            mlp_saturation_count.saturating_add(batch_result.mlp_saturation_count);
                        attention_saturation_count = attention_saturation_count
                            .saturating_add(batch_result.attention_saturation_count);
                        residual_saturation_count = residual_saturation_count
                            .saturating_add(batch_result.residual_saturation_count);
                        steps.extend(batch_result.steps);
                    }
                    Err(TrainError::CoreRejected(_)) => {
                        rejected_window_count =
                            rejected_window_count.saturating_add(batch_window_count);
                    }
                    Err(error) => return Err(error),
                }
            } else {
                for (relative_window_index, &window_start) in starts
                    [batch_start_index..batch_end_index]
                    .iter()
                    .enumerate()
                {
                    let window_index = batch_start_index + relative_window_index;
                    examined_windows += 1;
                    let target_token = tokens[window_start + config.seq_len];
                    let cache_before = match mini_transformer_forward_for_attention_and_position(
                        &model,
                        &tokens[window_start..window_start + config.seq_len],
                        config.attention_kind,
                        config.position_policy,
                    ) {
                        Ok(cache) => cache,
                        Err(_) => {
                            let mut recovered = None;
                            for checkpoint in rollback_history.iter().rev() {
                                if let Ok(cache) =
                                    mini_transformer_forward_for_attention_and_position(
                                        checkpoint,
                                        &tokens[window_start..window_start + config.seq_len],
                                        config.attention_kind,
                                        config.position_policy,
                                    )
                                {
                                    recovered = Some((checkpoint.clone(), cache));
                                    break;
                                }
                            }

                            match recovered {
                                Some((checkpoint, cache)) => {
                                    model = checkpoint;
                                    rollback_count = rollback_count.saturating_add(1);
                                    rejected_window_count = rejected_window_count.saturating_add(1);
                                    cache
                                }
                                None => {
                                    rejected_window_count = rejected_window_count.saturating_add(1);
                                    adaptive_attention_shifts.observe_rejected(
                                        rejected_batch_count.saturating_add(rejected_window_count),
                                        adaptive_shift_controller_enabled,
                                        config,
                                        &mut adaptive_shift_events,
                                    );
                                    continue;
                                }
                            }
                        }
                    };
                    let should_record_step = mini_transformer_should_record_step(
                        trace_detail,
                        updates.saturating_add(1),
                        trace_sample_interval,
                    );
                    let predicted_token_before = if should_record_step {
                        byte_argmax_i32(&cache_before.logits_q8)
                    } else {
                        0
                    };
                    let mut gradient_q15 = byte_vocab_softmax_gradient_q15(
                        &cache_before.probabilities_q15,
                        target_token,
                    );
                    apply_byte_argmax_margin_gradient_q15(
                        &mut gradient_q15,
                        &cache_before.logits_q8,
                        target_token,
                        config.argmax_margin_weight_q15,
                    );
                    let target_frequency_weight_q15 =
                        target_frequency_weights_q15[usize::from(target_token)];
                    let weighted_gradient_q15 =
                        byte_scale_gradient_q15(&gradient_q15, target_frequency_weight_q15);
                    let grad_output_q15 = byte_gradient_i32_to_i16(&weighted_gradient_q15);
                    let output_head_hash_before = if should_record_step {
                        model.output_head_hash()
                    } else {
                        0
                    };
                    let mlp_hash_before = if should_record_step {
                        model.mlp_hash()
                    } else {
                        0
                    };
                    let attention_hash_before = if should_record_step {
                        model.attention_hash()
                    } else {
                        0
                    };
                    let embedding_hash_before = if should_record_step {
                        model.embedding_hash()
                    } else {
                        0
                    };
                    let model_checkpoint = model.clone();
                    rollback_history.push(model_checkpoint.clone());
                    if rollback_history.len() > MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT {
                        rollback_history.remove(0);
                    }

                    if use_train_core_step {
                        let core_stats = {
                            let mut model_slices = nsrl_train_core::MiniTransformerModelSlicesMut {
                                embeddings: &mut model.embeddings,
                                q_weights: &mut model.q_weights,
                                k_weights: &mut model.k_weights,
                                v_weights: &mut model.v_weights,
                                o_weights: &mut model.o_weights,
                                up_weights: &mut model.up_weights,
                                gate_weights: &mut model.gate_weights,
                                down_weights: &mut model.down_weights,
                                output_weights: &mut model.output_weights,
                            };
                            let workspace_buffers = train_core_workspace
                                .as_mut()
                                .ok_or(TrainError::InvalidConfig)?;
                            let mut workspace = workspace_buffers.as_workspace();
                            nsrl_train_core::mini_transformer_linear_nope_train_step(
                                &mut model_slices,
                                &tokens[window_start..window_start + config.seq_len],
                                target_token,
                                nsrl_train_core::MiniTransformerStepConfig {
                                    seq_len: config.seq_len,
                                    learning_rate: config.learning_rate,
                                    output_learning_rate_shift: config.output_learning_rate_shift,
                                    mlp_learning_rate_shift: config.mlp_learning_rate_shift,
                                    embedding_learning_rate_shift: config
                                        .embedding_learning_rate_shift,
                                    attention_learning_rate_shift: config
                                        .attention_learning_rate_shift,
                                    attention_q_learning_rate_shift: config
                                        .attention_q_learning_rate_shift,
                                    attention_qk_learning_rate_shift: config
                                        .attention_qk_learning_rate_shift,
                                },
                                &mut workspace,
                            )
                        };
                        let core_stats = match core_stats {
                            Ok(stats) => stats,
                            Err(_) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                        };

                        let cache_after = match mini_transformer_forward_for_attention_and_position(
                            &model,
                            &tokens[window_start..window_start + config.seq_len],
                            config.attention_kind,
                            config.position_policy,
                        ) {
                            Ok(cache) => cache,
                            Err(_) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                        };

                        if mini_transformer_validate_guard_windows(
                            &model,
                            tokens,
                            &starts,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                            epoch,
                            window_index,
                            config.epochs,
                        )
                        .is_err()
                        {
                            model = model_checkpoint;
                            rollback_count = rollback_count.saturating_add(1);
                            rejected_window_count = rejected_window_count.saturating_add(1);
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count.saturating_add(rejected_window_count),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                            continue;
                        }

                        let predicted_token_after = if should_record_step {
                            byte_argmax_i32(&cache_after.logits_q8)
                        } else {
                            0
                        };
                        let output_head_hash_after = if should_record_step {
                            model.output_head_hash()
                        } else {
                            0
                        };
                        let mlp_hash_after = if should_record_step {
                            model.mlp_hash()
                        } else {
                            0
                        };
                        let attention_hash_after = if should_record_step {
                            model.attention_hash()
                        } else {
                            0
                        };
                        let embedding_hash_after = if should_record_step {
                            model.embedding_hash()
                        } else {
                            0
                        };

                        updates += 1;
                        output_head_saturation_count +=
                            core_stats.output_head.gradient_saturation_count;
                        output_head_zero_delta_count += core_stats.output_head.zero_delta_count;
                        output_head_delta_l1 = output_head_delta_l1
                            .saturating_add(core_stats.output_head.weight_delta_l1);
                        mlp_saturation_count += core_stats.mlp.gradient_saturation_count();
                        mlp_zero_delta_count += core_stats.mlp.zero_delta_count();
                        mlp_delta_l1 =
                            mlp_delta_l1.saturating_add(core_stats.mlp.weight_delta_l1());
                        embedding_saturation_count +=
                            core_stats.embedding.gradient_saturation_count;
                        embedding_zero_delta_count += core_stats.embedding.zero_delta_count;
                        embedding_delta_l1 =
                            embedding_delta_l1.saturating_add(core_stats.embedding.weight_delta_l1);
                        attention_saturation_count +=
                            core_stats.attention.gradient_saturation_count();
                        attention_zero_delta_count += core_stats.attention.zero_delta_count();
                        attention_delta_l1 = attention_delta_l1
                            .saturating_add(core_stats.attention.weight_delta_l1());
                        attention_q_delta_l1 = attention_q_delta_l1
                            .saturating_add(core_stats.attention.q.weight_delta_l1);
                        attention_k_delta_l1 = attention_k_delta_l1
                            .saturating_add(core_stats.attention.k.weight_delta_l1);
                        attention_v_delta_l1 = attention_v_delta_l1
                            .saturating_add(core_stats.attention.v.weight_delta_l1);
                        attention_o_delta_l1 = attention_o_delta_l1
                            .saturating_add(core_stats.attention.o.weight_delta_l1);
                        residual_saturation_count = residual_saturation_count
                            .saturating_add(core_stats.residual_saturation_count);

                        if should_record_step {
                            steps.push(MiniTransformerMlpTrainingStepTrace {
                                update_index: updates,
                                epoch,
                                window_index,
                                window_start,
                                first_token: tokens[window_start],
                                last_token: tokens[window_start + config.seq_len - 1],
                                target_token,
                                predicted_token_before,
                                predicted_token_after,
                                target_probability_before_q15: cache_before.probabilities_q15
                                    [usize::from(target_token)],
                                target_probability_after_q15: cache_after.probabilities_q15
                                    [usize::from(target_token)],
                                embedding_cache_hash: hash_i16_slice(
                                    &cache_before.embedding_output,
                                ),
                                attention_cache_hash: hash_i16_slice(
                                    &cache_before.attention_output,
                                ),
                                mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                                block_output_hash_before: hash_i16_slice(
                                    &cache_before.block_output,
                                ),
                                block_output_hash_after: hash_i16_slice(&cache_after.block_output),
                                output_head_hash_before,
                                output_head_hash_after,
                                mlp_hash_before,
                                mlp_hash_after,
                                attention_hash_before,
                                attention_hash_after,
                                embedding_hash_before,
                                embedding_hash_after,
                                output_head_saturation_count: core_stats
                                    .output_head
                                    .gradient_saturation_count,
                                mlp_saturation_count: core_stats.mlp.gradient_saturation_count(),
                                embedding_saturation_count: core_stats
                                    .embedding
                                    .gradient_saturation_count,
                                attention_saturation_count: core_stats
                                    .attention
                                    .gradient_saturation_count(),
                                residual_saturation_count: core_stats.residual_saturation_count,
                                output_head_zero_delta_count: core_stats
                                    .output_head
                                    .zero_delta_count,
                                mlp_zero_delta_count: core_stats.mlp.zero_delta_count(),
                                embedding_zero_delta_count: core_stats.embedding.zero_delta_count,
                                attention_zero_delta_count: core_stats.attention.zero_delta_count(),
                                output_head_delta_l1: core_stats.output_head.weight_delta_l1,
                                mlp_delta_l1: core_stats.mlp.weight_delta_l1(),
                                embedding_delta_l1: core_stats.embedding.weight_delta_l1,
                                attention_delta_l1: core_stats.attention.weight_delta_l1(),
                                attention_q_delta_l1: core_stats.attention.q.weight_delta_l1,
                                attention_k_delta_l1: core_stats.attention.k.weight_delta_l1,
                                attention_v_delta_l1: core_stats.attention.v.weight_delta_l1,
                                attention_o_delta_l1: core_stats.attention.o.weight_delta_l1,
                            });
                        }
                        continue;
                    }

                    let workspace = host_training_workspace
                        .as_mut()
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.reset_host_training_step();
                    linear_backward_input_i16_i8_i16_per_channel_checked(
                        &grad_output_q15,
                        LinearBackwardInputI16I8Params {
                            weights: &model.output_weights,
                            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                            grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
                            input_dim: MINI_TRANSFORMER_D_MODEL,
                            output_dim: BYTE_VOCAB,
                        },
                        LinearBackwardInputWorkspace {
                            scaled_grad_output: &mut workspace.output_scaled_grad,
                        },
                        &mut workspace.grad_last_features,
                    )
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_output_head_backward_input",
                    ))?;

                    let last_start = (config.seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
                    let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
                    let runtime_config = adaptive_attention_shifts.runtime_config(config);
                    let output_update = if use_output_head_accumulator {
                        empty_linear_weight_update_stats()
                    } else {
                        linear_backward_weight_update_i8_checked(
                            &cache_before.output_features,
                            &grad_output_q15,
                            &mut model.output_weights,
                            LinearBackwardWeightUpdateI8Params {
                                forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                                input_dim: MINI_TRANSFORMER_D_MODEL,
                                output_dim: BYTE_VOCAB,
                                learning_rate: config.learning_rate,
                                learning_rate_shift: runtime_config.output_learning_rate_shift,
                            },
                            LinearBackwardWeightUpdateWorkspace {
                                scaled_grad_output: &mut workspace.output_scaled_grad,
                            },
                        )
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_output_head_update",
                        ))?
                    };

                    let total = config
                        .seq_len
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .ok_or(TrainError::InvalidConfig)?;
                    let (
                        mlp_input_saturation_count,
                        gradient_residual_saturation_count,
                        mlp_update,
                        attention_update,
                        embedding_gradient_saturation_count,
                        embedding_update,
                    ) = if use_stacked_serial_backprop {
                        let mut grad_block_output = vec![0_i16; total];
                        grad_block_output[last_start..last_end]
                            .copy_from_slice(&workspace.grad_last_features);
                        let mut stacked_mlp_update = empty_gated_mlp_weight_update_stats();
                        let mut stacked_attention_update =
                            empty_mini_transformer_attention_weight_update_stats();
                        let mut stacked_mlp_input_saturation_count = 0_usize;
                        let mut stacked_gradient_residual_saturation_count = 0_usize;
                        let mut stacked_input_gradient_saturation_count = 0_usize;
                        let mut stacked_rejected = false;

                        for layer_index in (0..cache_before.layers.len()).rev() {
                            let layer_runtime_config =
                                mini_transformer_stacked_layer_runtime_config(
                                    runtime_config,
                                    layer_index,
                                    cache_before.layers.len(),
                                );
                            let block_update =
                                match mini_transformer_block_backward_update_i8_checked(
                                    &cache_before.layers[layer_index],
                                    &grad_block_output,
                                    &mut model,
                                    layer_index,
                                    layer_runtime_config,
                                    workspace,
                                ) {
                                    Ok(update) => update,
                                    Err(TrainError::CoreRejected(_)) => {
                                        model = model_checkpoint.clone();
                                        rollback_count = rollback_count.saturating_add(1);
                                        rejected_window_count =
                                            rejected_window_count.saturating_add(1);
                                        adaptive_attention_shifts.observe_rejected(
                                            rejected_batch_count
                                                .saturating_add(rejected_window_count),
                                            adaptive_shift_controller_enabled,
                                            config,
                                            &mut adaptive_shift_events,
                                        );
                                        stacked_rejected = true;
                                        break;
                                    }
                                    Err(error) => return Err(error),
                                };
                            add_gated_mlp_weight_update_stats_checked(
                                &mut stacked_mlp_update,
                                block_update.mlp_update,
                            )?;
                            add_mini_transformer_attention_weight_update_stats_checked(
                                &mut stacked_attention_update,
                                block_update.attention_update,
                            )?;
                            stacked_mlp_input_saturation_count = stacked_mlp_input_saturation_count
                                .saturating_add(block_update.mlp_input_saturation_count);
                            stacked_gradient_residual_saturation_count =
                                stacked_gradient_residual_saturation_count.saturating_add(
                                    block_update.gradient_residual_saturation_count,
                                );
                            stacked_input_gradient_saturation_count =
                                stacked_input_gradient_saturation_count
                                    .saturating_add(block_update.input_gradient_saturation_count);
                            grad_block_output = block_update.grad_input;
                        }
                        if stacked_rejected {
                            continue;
                        }

                        workspace.grad_embedding_output[..total]
                            .copy_from_slice(&grad_block_output);
                        let stacked_embedding_learning_rate_shift = runtime_config
                            .embedding_learning_rate_shift
                            .saturating_add(
                                MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT,
                            )
                            .min(MAX_RIGHT_SHIFT);
                        let embedding_update =
                            apply_mini_transformer_embedding_update_with_position_policy(
                                &mut model.embeddings,
                                &mut model.position_embeddings,
                                &tokens[window_start..window_start + config.seq_len],
                                &workspace.grad_embedding_output,
                                config.position_policy,
                                config.learning_rate,
                                stacked_embedding_learning_rate_shift,
                            )?;
                        (
                            stacked_mlp_input_saturation_count,
                            stacked_gradient_residual_saturation_count,
                            stacked_mlp_update,
                            stacked_attention_update,
                            stacked_input_gradient_saturation_count,
                            embedding_update,
                        )
                    } else {
                        workspace.grad_mlp_output[last_start..last_end]
                            .copy_from_slice(&workspace.grad_last_features);
                        let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
                            &workspace.grad_mlp_output,
                            mini_transformer_final_mlp_params(&model, config.seq_len)?,
                            &cache_before.mlp_up,
                            &cache_before.mlp_gate,
                            GatedMlpBackwardScales {
                                down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                                gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                            },
                            GatedMlpBackwardWorkspace {
                                scaled_grad_output: &mut workspace.mlp_scaled_grad,
                                grad_gated: &mut workspace.mlp_input_grad_gated,
                                grad_up: &mut workspace.mlp_input_grad_up,
                                grad_gate: &mut workspace.mlp_input_grad_gate,
                                grad_up_input: &mut workspace.mlp_input_grad_up_input,
                                grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
                            },
                            &mut workspace.grad_mlp_input,
                        )
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_mlp_backward_input",
                        ))?;

                        let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                            &workspace.grad_mlp_output,
                            &workspace.grad_mlp_input,
                            &mut workspace.grad_attention_output,
                        )?;

                        let mlp_update = if use_mlp_accumulator {
                            empty_gated_mlp_weight_update_stats()
                        } else {
                            let up_or_gate_range = model.final_mlp_up_or_gate_weight_range()?;
                            let down_range = model.final_mlp_down_weight_range()?;
                            gated_mlp_backward_weight_update_i8_checked(
                                &cache_before.mlp_norm,
                                &workspace.grad_mlp_output,
                                &cache_before.mlp_up,
                                &cache_before.mlp_gate,
                                &cache_before.mlp_gated,
                                &mut model.up_weights[up_or_gate_range.clone()],
                                &mut model.gate_weights[up_or_gate_range],
                                &mut model.down_weights[down_range],
                                GatedMlpWeightUpdateParams {
                                    up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                    gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                    down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                                    down_to_hidden_scales:
                                        &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                    seq_len: config.seq_len,
                                    d_model: MINI_TRANSFORMER_D_MODEL,
                                    hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                                    learning_rate: config.learning_rate,
                                    learning_rate_shift: runtime_config.mlp_learning_rate_shift,
                                },
                                GatedMlpWeightUpdateWorkspace {
                                    scaled_grad_output: &mut workspace.mlp_scaled_grad,
                                    grad_gated: &mut workspace.mlp_update_grad_gated,
                                    grad_up: &mut workspace.mlp_update_grad_up,
                                    grad_gate: &mut workspace.mlp_update_grad_gate,
                                },
                            )
                            .ok_or(TrainError::CoreRejected("mini_transformer_mlp_update"))?
                        };

                        let attention_update = match mini_transformer_attention_update_i8_checked(
                            cache_before
                                .layers
                                .last()
                                .ok_or(TrainError::InvalidConfig)?,
                            &mut model,
                            transformer_layers - 1,
                            runtime_config,
                            workspace,
                            if use_attention_accumulator {
                                Some(&mut attention_weight_gradients[transformer_layers - 1])
                            } else {
                                None
                            },
                        ) {
                            Ok(update) => update,
                            Err(TrainError::CoreRejected(_)) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                            Err(error) => return Err(error),
                        };

                        let embedding_gradient_saturation_count = add_i16_residual_rows_checked(
                            &workspace.grad_attention_output,
                            &workspace.grad_attention_norm_input,
                            &mut workspace.grad_embedding_output,
                        )?;
                        let embedding_update = if use_embedding_accumulator {
                            empty_softmax_update_stats()
                        } else {
                            apply_mini_transformer_embedding_update_with_position_policy(
                                &mut model.embeddings,
                                &mut model.position_embeddings,
                                &tokens[window_start..window_start + config.seq_len],
                                &workspace.grad_embedding_output,
                                config.position_policy,
                                config.learning_rate,
                                runtime_config.embedding_learning_rate_shift,
                            )?
                        };

                        (
                            mlp_input_saturation_count,
                            gradient_residual_saturation_count,
                            mlp_update,
                            attention_update,
                            embedding_gradient_saturation_count,
                            embedding_update,
                        )
                    };
                    let mlp_rms_backward_saturation_count = 0_usize;
                    let attention_rms_backward_saturation_count = 0_usize;

                    let cache_after = match mini_transformer_forward_for_attention_and_position(
                        &model,
                        &tokens[window_start..window_start + config.seq_len],
                        config.attention_kind,
                        config.position_policy,
                    ) {
                        Ok(cache) => cache,
                        Err(error) => {
                            let _ = error;
                            model = model_checkpoint;
                            rollback_count = rollback_count.saturating_add(1);
                            rejected_window_count = rejected_window_count.saturating_add(1);
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count.saturating_add(rejected_window_count),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                            continue;
                        }
                    };

                    if mini_transformer_validate_guard_windows(
                        &model,
                        tokens,
                        &starts,
                        config.seq_len,
                        config.attention_kind,
                        config.position_policy,
                        epoch,
                        window_index,
                        config.epochs,
                    )
                    .is_err()
                    {
                        model = model_checkpoint;
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count = rejected_window_count.saturating_add(1);
                        adaptive_attention_shifts.observe_rejected(
                            rejected_batch_count.saturating_add(rejected_window_count),
                            adaptive_shift_controller_enabled,
                            config,
                            &mut adaptive_shift_events,
                        );
                        continue;
                    }
                    if use_output_head_accumulator {
                        accumulate_linear_weight_gradient_i64_prescaled(
                            &cache_before.output_features,
                            &workspace.output_scaled_grad,
                            &mut output_head_gradient,
                        )?;
                    }
                    if use_mlp_accumulator {
                        accumulate_gated_mlp_weight_gradient_i64(
                            &cache_before.mlp_norm,
                            &workspace.grad_mlp_output,
                            &cache_before.mlp_gated,
                            &workspace.mlp_input_grad_up,
                            &workspace.mlp_input_grad_gate,
                            GatedMlpWeightUpdateParams {
                                up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                                down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                seq_len: config.seq_len,
                                d_model: MINI_TRANSFORMER_D_MODEL,
                                hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                                learning_rate: config.learning_rate,
                                learning_rate_shift: config.mlp_learning_rate_shift,
                            },
                            &mut mlp_weight_gradients[transformer_layers - 1],
                            &mut workspace.mlp_scaled_grad,
                        )?;
                    }
                    if use_embedding_accumulator {
                        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                            &tokens[window_start..window_start + config.seq_len],
                            &workspace.grad_embedding_output,
                            config.position_policy,
                            &mut embedding_gradient,
                        )?;
                    }
                    let predicted_token_after = if should_record_step {
                        byte_argmax_i32(&cache_after.logits_q8)
                    } else {
                        0
                    };
                    let output_head_hash_after = if should_record_step {
                        model.output_head_hash()
                    } else {
                        0
                    };
                    let mlp_hash_after = if should_record_step {
                        model.mlp_hash()
                    } else {
                        0
                    };
                    let attention_hash_after = if should_record_step {
                        model.attention_hash()
                    } else {
                        0
                    };
                    let embedding_hash_after = if should_record_step {
                        model.embedding_hash()
                    } else {
                        0
                    };

                    updates += 1;
                    output_head_saturation_count += output_update.gradient_saturation_count;
                    output_head_zero_delta_count += output_update.zero_delta_count;
                    output_head_delta_l1 =
                        output_head_delta_l1.saturating_add(output_update.weight_delta_l1);
                    mlp_saturation_count += mlp_input_saturation_count;
                    mlp_saturation_count += mlp_rms_backward_saturation_count;
                    mlp_saturation_count +=
                        mlp_update.gradient_saturation_count().unwrap_or(usize::MAX);
                    mlp_zero_delta_count += mlp_update.zero_delta_count().unwrap_or(usize::MAX);
                    mlp_delta_l1 =
                        mlp_delta_l1.saturating_add(mlp_update.weight_delta_l1().unwrap_or(0));
                    embedding_saturation_count += embedding_update.gradient_saturation_count;
                    embedding_zero_delta_count += embedding_update.zero_delta_count;
                    embedding_delta_l1 =
                        embedding_delta_l1.saturating_add(embedding_update.weight_delta_l1);
                    attention_saturation_count += attention_update.gradient_saturation_count;
                    attention_saturation_count += attention_rms_backward_saturation_count;
                    attention_zero_delta_count += attention_update.zero_delta_count;
                    attention_delta_l1 =
                        attention_delta_l1.saturating_add(attention_update.weight_delta_l1);
                    attention_q_delta_l1 =
                        attention_q_delta_l1.saturating_add(attention_update.q.weight_delta_l1);
                    attention_k_delta_l1 =
                        attention_k_delta_l1.saturating_add(attention_update.k.weight_delta_l1);
                    attention_v_delta_l1 =
                        attention_v_delta_l1.saturating_add(attention_update.v.weight_delta_l1);
                    attention_o_delta_l1 =
                        attention_o_delta_l1.saturating_add(attention_update.o.weight_delta_l1);
                    residual_saturation_count += gradient_residual_saturation_count;
                    residual_saturation_count += embedding_gradient_saturation_count;
                    residual_saturation_count += cache_before.residual_saturation_count;
                    residual_saturation_count += cache_after.residual_saturation_count;
                    if !use_attention_accumulator {
                        adaptive_attention_shifts.observe_accepted(
                            output_update,
                            mlp_update,
                            embedding_update,
                            &attention_update,
                            accepted_batch_count.saturating_add(updates),
                            adaptive_shift_controller_enabled,
                            config,
                            &mut adaptive_shift_events,
                        );
                    }

                    if should_record_step {
                        steps.push(MiniTransformerMlpTrainingStepTrace {
                            update_index: updates,
                            epoch,
                            window_index,
                            window_start,
                            first_token: tokens[window_start],
                            last_token: tokens[window_start + config.seq_len - 1],
                            target_token,
                            predicted_token_before,
                            predicted_token_after,
                            target_probability_before_q15: cache_before.probabilities_q15
                                [usize::from(target_token)],
                            target_probability_after_q15: cache_after.probabilities_q15
                                [usize::from(target_token)],
                            embedding_cache_hash: hash_i16_slice(&cache_before.embedding_output),
                            attention_cache_hash: hash_i16_slice(&cache_before.attention_output),
                            mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                            block_output_hash_before: hash_i16_slice(&cache_before.block_output),
                            block_output_hash_after: hash_i16_slice(&cache_after.block_output),
                            output_head_hash_before,
                            output_head_hash_after,
                            mlp_hash_before,
                            mlp_hash_after,
                            attention_hash_before,
                            attention_hash_after,
                            embedding_hash_before,
                            embedding_hash_after,
                            output_head_saturation_count: output_update.gradient_saturation_count,
                            mlp_saturation_count: mlp_input_saturation_count
                                + mlp_rms_backward_saturation_count
                                + mlp_update.gradient_saturation_count().unwrap_or(usize::MAX),
                            embedding_saturation_count: embedding_update.gradient_saturation_count,
                            attention_saturation_count: attention_update.gradient_saturation_count
                                + attention_rms_backward_saturation_count,
                            residual_saturation_count: cache_before.residual_saturation_count
                                + cache_after.residual_saturation_count
                                + gradient_residual_saturation_count
                                + embedding_gradient_saturation_count,
                            output_head_zero_delta_count: output_update.zero_delta_count,
                            mlp_zero_delta_count: mlp_update
                                .zero_delta_count()
                                .unwrap_or(usize::MAX),
                            embedding_zero_delta_count: embedding_update.zero_delta_count,
                            attention_zero_delta_count: attention_update.zero_delta_count,
                            output_head_delta_l1: output_update.weight_delta_l1,
                            mlp_delta_l1: mlp_update.weight_delta_l1().unwrap_or(0),
                            embedding_delta_l1: embedding_update.weight_delta_l1,
                            attention_delta_l1: attention_update.weight_delta_l1,
                            attention_q_delta_l1: attention_update.q.weight_delta_l1,
                            attention_k_delta_l1: attention_update.k.weight_delta_l1,
                            attention_v_delta_l1: attention_update.v.weight_delta_l1,
                            attention_o_delta_l1: attention_update.o.weight_delta_l1,
                        });
                    }
                }
            }

            let accepted_windows_in_batch = updates.saturating_sub(updates_before_batch);
            if accepted_windows_in_batch > 0 {
                if rollback_count == rollbacks_before_batch {
                    let mut candidate_model = model.clone();
                    average_mini_transformer_batch_movement(
                        &batch_model_checkpoint,
                        &mut candidate_model,
                        accepted_windows_in_batch,
                        !use_output_head_accumulator,
                        !use_mlp_accumulator,
                        !use_attention_accumulator,
                        !use_embedding_accumulator,
                    )?;

                    let mut batch_output_head_saturation_count = 0_usize;
                    let mut batch_output_head_zero_delta_count = 0_usize;
                    let mut batch_output_head_delta_l1 = 0_u64;
                    let mut batch_mlp_saturation_count = 0_usize;
                    let mut batch_mlp_zero_delta_count = 0_usize;
                    let mut batch_mlp_delta_l1 = 0_u64;
                    let mut batch_attention_saturation_count = 0_usize;
                    let mut batch_attention_zero_delta_count = 0_usize;
                    let mut batch_attention_delta_l1 = 0_u64;
                    let mut batch_attention_q_delta_l1 = 0_u64;
                    let mut batch_attention_k_delta_l1 = 0_u64;
                    let mut batch_attention_v_delta_l1 = 0_u64;
                    let mut batch_attention_o_delta_l1 = 0_u64;
                    let mut batch_attention_update_for_controller = None;
                    let mut batch_embedding_saturation_count = 0_usize;
                    let mut batch_embedding_zero_delta_count = 0_usize;
                    let mut batch_embedding_delta_l1 = 0_u64;

                    let mut batch_output_head_accumulator_batch_count = 0_usize;
                    let mut batch_output_head_accumulator_window_count = 0_usize;
                    let mut batch_mlp_accumulator_batch_count = 0_usize;
                    let mut batch_mlp_accumulator_window_count = 0_usize;
                    let mut batch_attention_accumulator_batch_count = 0_usize;
                    let mut batch_attention_accumulator_window_count = 0_usize;
                    let mut batch_embedding_accumulator_batch_count = 0_usize;
                    let mut batch_embedding_accumulator_window_count = 0_usize;

                    let output_head_gradient_checkpoint = output_head_gradient.clone();
                    let mlp_weight_gradients_checkpoint = mlp_weight_gradients.clone();
                    let attention_weight_gradients_checkpoint = attention_weight_gradients.clone();
                    let embedding_gradient_checkpoint = embedding_gradient.clone();
                    let batch_windows = &starts[batch_start_index..batch_end_index];
                    let batch_runtime_config = adaptive_attention_shifts.runtime_config(config);
                    let batch_apply_config = mini_transformer_batch_component_shift_config(
                        batch_runtime_config,
                        accepted_windows_in_batch,
                    )?;

                    if use_output_head_accumulator {
                        let output_batch_update = apply_linear_weight_gradient_i64_to_i8(
                            &mut output_head_gradient,
                            &mut candidate_model.output_weights,
                            batch_apply_config.learning_rate,
                            batch_apply_config.output_learning_rate_shift,
                            true,
                        )?;
                        batch_output_head_saturation_count =
                            output_batch_update.gradient_saturation_count;
                        batch_output_head_zero_delta_count = output_batch_update.zero_delta_count;
                        batch_output_head_delta_l1 = output_batch_update.weight_delta_l1;
                        batch_output_head_accumulator_batch_count = 1;
                        batch_output_head_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_mlp_accumulator {
                        let mut mlp_batch_update = empty_gated_mlp_weight_update_stats();
                        for (layer_index, gradient) in mlp_weight_gradients
                            .iter_mut()
                            .enumerate()
                            .take(transformer_layers)
                        {
                            let layer_apply_config = if transformer_layers > 1 {
                                mini_transformer_stacked_layer_runtime_config(
                                    batch_apply_config,
                                    layer_index,
                                    transformer_layers,
                                )
                            } else {
                                batch_apply_config
                            };
                            let up_or_gate_range =
                                candidate_model.mlp_up_or_gate_weight_range(layer_index)?;
                            let down_range = candidate_model.mlp_down_weight_range(layer_index)?;
                            let layer_update = apply_gated_mlp_weight_gradient_i64_to_i8(
                                gradient,
                                &mut candidate_model.up_weights[up_or_gate_range.clone()],
                                &mut candidate_model.gate_weights[up_or_gate_range],
                                &mut candidate_model.down_weights[down_range],
                                layer_apply_config.learning_rate,
                                layer_apply_config.mlp_learning_rate_shift,
                                true,
                            )?;
                            add_gated_mlp_weight_update_stats_checked(
                                &mut mlp_batch_update,
                                layer_update,
                            )?;
                        }
                        batch_mlp_saturation_count = mlp_batch_update
                            .gradient_saturation_count()
                            .unwrap_or(usize::MAX);
                        batch_mlp_zero_delta_count =
                            mlp_batch_update.zero_delta_count().unwrap_or(usize::MAX);
                        batch_mlp_delta_l1 = mlp_batch_update.weight_delta_l1().unwrap_or(0);
                        batch_mlp_accumulator_batch_count = 1;
                        batch_mlp_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_attention_accumulator {
                        let mut attention_batch_update =
                            empty_mini_transformer_attention_weight_update_stats();
                        for (layer_index, gradient) in attention_weight_gradients
                            .iter_mut()
                            .enumerate()
                            .take(transformer_layers)
                        {
                            let layer_apply_config = if transformer_layers > 1 {
                                mini_transformer_stacked_layer_runtime_config(
                                    batch_apply_config,
                                    layer_index,
                                    transformer_layers,
                                )
                            } else {
                                batch_apply_config
                            };
                            let layer_update = if transformer_layers == 1 {
                                apply_mini_transformer_attention_weight_gradient_i64_to_i8(
                                    gradient,
                                    &mut candidate_model,
                                    layer_apply_config,
                                )?
                            } else {
                                apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
                                    gradient,
                                    &mut candidate_model,
                                    layer_index,
                                    layer_apply_config,
                                )?
                            };
                            add_mini_transformer_attention_weight_update_stats_checked(
                                &mut attention_batch_update,
                                layer_update,
                            )?;
                        }
                        if config.attention_vo_oracle {
                            let (v_oracle, o_oracle) =
                                mini_transformer_attention_vo_oracle_update_i8_checked(
                                    &mut candidate_model,
                                    tokens,
                                    batch_windows,
                                    config.seq_len,
                                    config.learning_rate,
                                )?;
                            add_linear_weight_update_stats_checked(
                                &mut attention_batch_update.v,
                                v_oracle,
                            )?;
                            add_linear_weight_update_stats_checked(
                                &mut attention_batch_update.o,
                                o_oracle,
                            )?;
                            attention_batch_update.gradient_saturation_count =
                                attention_batch_update
                                    .gradient_saturation_count
                                    .saturating_add(v_oracle.gradient_saturation_count)
                                    .saturating_add(o_oracle.gradient_saturation_count);
                            attention_batch_update.zero_delta_count = attention_batch_update
                                .zero_delta_count
                                .saturating_add(v_oracle.zero_delta_count)
                                .saturating_add(o_oracle.zero_delta_count);
                            attention_batch_update.weight_delta_l1 = attention_batch_update
                                .weight_delta_l1
                                .saturating_add(v_oracle.weight_delta_l1)
                                .saturating_add(o_oracle.weight_delta_l1);
                        }
                        batch_attention_saturation_count =
                            attention_batch_update.gradient_saturation_count;
                        batch_attention_zero_delta_count = attention_batch_update.zero_delta_count;
                        batch_attention_delta_l1 = attention_batch_update.weight_delta_l1;
                        batch_attention_q_delta_l1 = attention_batch_update.q.weight_delta_l1;
                        batch_attention_k_delta_l1 = attention_batch_update.k.weight_delta_l1;
                        batch_attention_v_delta_l1 = attention_batch_update.v.weight_delta_l1;
                        batch_attention_o_delta_l1 = attention_batch_update.o.weight_delta_l1;
                        batch_attention_update_for_controller = Some(attention_batch_update);
                        batch_attention_accumulator_batch_count = 1;
                        batch_attention_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_embedding_accumulator {
                        let embedding_learning_rate_shift = if transformer_layers > 1 {
                            batch_apply_config
                                .embedding_learning_rate_shift
                                .saturating_add(
                                    MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT,
                                )
                                .min(MAX_RIGHT_SHIFT)
                        } else {
                            batch_apply_config.embedding_learning_rate_shift
                        };
                        let embedding_batch_update =
                            apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
                                &mut embedding_gradient,
                                &mut candidate_model.embeddings,
                                &mut candidate_model.position_embeddings,
                                config.position_policy,
                                batch_apply_config.learning_rate,
                                embedding_learning_rate_shift,
                            )?;
                        batch_embedding_saturation_count =
                            embedding_batch_update.gradient_saturation_count;
                        batch_embedding_zero_delta_count = embedding_batch_update.zero_delta_count;
                        batch_embedding_delta_l1 = embedding_batch_update.weight_delta_l1;
                        batch_embedding_accumulator_batch_count = 1;
                        batch_embedding_accumulator_window_count = accepted_windows_in_batch;
                    }

                    let batch_valid = mini_transformer_validate_batch_windows(
                        &candidate_model,
                        tokens,
                        batch_windows,
                        config.seq_len,
                        config.attention_kind,
                        config.position_policy,
                    )
                    .and_then(|_| {
                        mini_transformer_validate_guard_windows(
                            &candidate_model,
                            tokens,
                            &starts,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                            epoch,
                            batch_end_index.saturating_sub(1).min(starts.len() - 1),
                            config.epochs,
                        )
                    })
                    .is_ok();
                    let mut batch_loss_regressed = false;
                    if batch_valid && config.reject_loss_regression {
                        let loss_guard_starts = mini_transformer_loss_guard_starts(
                            &starts,
                            batch_start_index,
                            batch_end_index,
                        );
                        let before_loss = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                            tokens,
                            &loss_guard_starts,
                            &batch_model_checkpoint,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                        )?;
                        let after_loss = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                            tokens,
                            &loss_guard_starts,
                            &candidate_model,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                        );
                        batch_loss_regressed = match after_loss {
                            Ok(after_loss) => mini_transformer_loss_guard_regressed(
                                before_loss,
                                after_loss,
                                loss_guard_starts.len(),
                            ),
                            Err(TrainError::CoreRejected(_)) => true,
                            Err(error) => return Err(error),
                        };
                    }

                    if batch_valid && !batch_loss_regressed {
                        model = candidate_model;
                        output_head_saturation_count = output_head_saturation_count
                            .saturating_add(batch_output_head_saturation_count);
                        output_head_zero_delta_count = output_head_zero_delta_count
                            .saturating_add(batch_output_head_zero_delta_count);
                        output_head_delta_l1 =
                            output_head_delta_l1.saturating_add(batch_output_head_delta_l1);
                        mlp_saturation_count =
                            mlp_saturation_count.saturating_add(batch_mlp_saturation_count);
                        mlp_zero_delta_count =
                            mlp_zero_delta_count.saturating_add(batch_mlp_zero_delta_count);
                        mlp_delta_l1 = mlp_delta_l1.saturating_add(batch_mlp_delta_l1);
                        attention_saturation_count = attention_saturation_count
                            .saturating_add(batch_attention_saturation_count);
                        attention_zero_delta_count = attention_zero_delta_count
                            .saturating_add(batch_attention_zero_delta_count);
                        attention_delta_l1 =
                            attention_delta_l1.saturating_add(batch_attention_delta_l1);
                        attention_q_delta_l1 =
                            attention_q_delta_l1.saturating_add(batch_attention_q_delta_l1);
                        attention_k_delta_l1 =
                            attention_k_delta_l1.saturating_add(batch_attention_k_delta_l1);
                        attention_v_delta_l1 =
                            attention_v_delta_l1.saturating_add(batch_attention_v_delta_l1);
                        attention_o_delta_l1 =
                            attention_o_delta_l1.saturating_add(batch_attention_o_delta_l1);
                        embedding_saturation_count = embedding_saturation_count
                            .saturating_add(batch_embedding_saturation_count);
                        embedding_zero_delta_count = embedding_zero_delta_count
                            .saturating_add(batch_embedding_zero_delta_count);
                        embedding_delta_l1 =
                            embedding_delta_l1.saturating_add(batch_embedding_delta_l1);
                        output_head_accumulator_batch_count = output_head_accumulator_batch_count
                            .saturating_add(batch_output_head_accumulator_batch_count);
                        output_head_accumulator_window_count = output_head_accumulator_window_count
                            .saturating_add(batch_output_head_accumulator_window_count);
                        mlp_accumulator_batch_count = mlp_accumulator_batch_count
                            .saturating_add(batch_mlp_accumulator_batch_count);
                        mlp_accumulator_window_count = mlp_accumulator_window_count
                            .saturating_add(batch_mlp_accumulator_window_count);
                        attention_accumulator_batch_count = attention_accumulator_batch_count
                            .saturating_add(batch_attention_accumulator_batch_count);
                        attention_accumulator_window_count = attention_accumulator_window_count
                            .saturating_add(batch_attention_accumulator_window_count);
                        embedding_accumulator_batch_count = embedding_accumulator_batch_count
                            .saturating_add(batch_embedding_accumulator_batch_count);
                        embedding_accumulator_window_count = embedding_accumulator_window_count
                            .saturating_add(batch_embedding_accumulator_window_count);
                        if let Some(update) = batch_attention_update_for_controller.as_ref() {
                            adaptive_attention_shifts.observe_accepted(
                                LinearWeightUpdateStats {
                                    gradient_saturation_count: batch_output_head_saturation_count,
                                    zero_delta_count: batch_output_head_zero_delta_count,
                                    weight_delta_l1: batch_output_head_delta_l1,
                                },
                                GatedMlpWeightUpdateStats {
                                    down: LinearWeightUpdateStats {
                                        gradient_saturation_count: batch_mlp_saturation_count,
                                        zero_delta_count: batch_mlp_zero_delta_count,
                                        weight_delta_l1: batch_mlp_delta_l1,
                                    },
                                    up: empty_linear_weight_update_stats(),
                                    gate: empty_linear_weight_update_stats(),
                                },
                                SoftmaxUpdateStats {
                                    gradient_saturation_count: batch_embedding_saturation_count,
                                    zero_delta_count: batch_embedding_zero_delta_count,
                                    weight_delta_l1: batch_embedding_delta_l1,
                                },
                                update,
                                accepted_batch_count.saturating_add(1),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                        }
                        accepted_batch_count = accepted_batch_count.saturating_add(1);
                        emit_mini_transformer_committed_binary_steps(
                            &steps,
                            steps_before_batch,
                            &mut binary_trace,
                        )?;
                    } else {
                        model = batch_model_checkpoint;
                        updates = updates_before_batch;
                        steps.truncate(steps_before_batch);
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count =
                            rejected_window_count.saturating_add(accepted_windows_in_batch);
                        rejected_batch_count = rejected_batch_count.saturating_add(1);
                        if batch_loss_regressed {
                            loss_regression_rejected_batch_count =
                                loss_regression_rejected_batch_count.saturating_add(1);
                        } else {
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count,
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                        }
                        if use_output_head_accumulator {
                            output_head_gradient = output_head_gradient_checkpoint;
                            output_head_gradient.clear();
                        }
                        if use_mlp_accumulator {
                            mlp_weight_gradients = mlp_weight_gradients_checkpoint;
                            mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                                &mut mlp_weight_gradients,
                            );
                        }
                        if use_attention_accumulator {
                            attention_weight_gradients = attention_weight_gradients_checkpoint;
                            mini_transformer_clear_attention_weight_gradient_i64_layers(
                                &mut attention_weight_gradients,
                            );
                        }
                        if use_embedding_accumulator {
                            embedding_gradient = embedding_gradient_checkpoint;
                            embedding_gradient.clear();
                        }
                    }
                } else {
                    if use_output_head_accumulator {
                        output_head_gradient.clear();
                    }
                    if use_mlp_accumulator {
                        mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                            &mut mlp_weight_gradients,
                        );
                    }
                    if use_attention_accumulator {
                        mini_transformer_clear_attention_weight_gradient_i64_layers(
                            &mut attention_weight_gradients,
                        );
                    }
                    if use_embedding_accumulator {
                        embedding_gradient.clear();
                    }
                    accepted_batch_count = accepted_batch_count.saturating_add(1);
                    emit_mini_transformer_committed_binary_steps(
                        &steps,
                        steps_before_batch,
                        &mut binary_trace,
                    )?;
                }
            } else {
                if use_output_head_accumulator {
                    output_head_gradient.clear();
                }
                if use_mlp_accumulator {
                    mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                        &mut mlp_weight_gradients,
                    );
                }
                if use_attention_accumulator {
                    mini_transformer_clear_attention_weight_gradient_i64_layers(
                        &mut attention_weight_gradients,
                    );
                }
                if use_embedding_accumulator {
                    embedding_gradient.clear();
                }
                rejected_batch_count = rejected_batch_count.saturating_add(1);
                adaptive_attention_shifts.observe_rejected(
                    rejected_batch_count,
                    adaptive_shift_controller_enabled,
                    config,
                    &mut adaptive_shift_events,
                );
            }
            output_head_carry_l1 = if use_output_head_accumulator {
                output_head_gradient.residual_l1()
            } else {
                0
            };
            mlp_carry_l1 = if use_mlp_accumulator {
                mini_transformer_gated_mlp_weight_gradient_i64_layers_residual_l1(
                    &mlp_weight_gradients,
                )
            } else {
                0
            };
            embedding_carry_l1 = if use_embedding_accumulator {
                embedding_gradient.residual_l1(config.position_policy)
            } else {
                0
            };
            if use_attention_accumulator {
                attention_q_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Query,
                    );
                attention_k_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Key,
                    );
                attention_v_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Value,
                    );
                attention_o_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Output,
                    );
                attention_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_residual_l1(
                        &attention_weight_gradients,
                    );
            } else {
                attention_q_carry_l1 = 0;
                attention_k_carry_l1 = 0;
                attention_v_carry_l1 = 0;
                attention_o_carry_l1 = 0;
                attention_carry_l1 = 0;
            }
            let observed_batch_count = accepted_batch_count.saturating_add(rejected_batch_count);
            if progress_interval_batches > 0
                && observed_batch_count > 0
                && observed_batch_count.is_multiple_of(progress_interval_batches)
            {
                progress(&mini_transformer_training_progress_trace(
                    config,
                    tokens.len(),
                    token_hash,
                    window_hash,
                    starts.len(),
                    examined_windows,
                    updates,
                    accepted_batch_count,
                    rejected_batch_count,
                    rollback_count,
                    rejected_window_count,
                    output_head_delta_l1,
                    mlp_delta_l1,
                    embedding_delta_l1,
                    attention_delta_l1,
                    attention_q_delta_l1,
                    attention_k_delta_l1,
                    attention_v_delta_l1,
                    attention_o_delta_l1,
                    output_head_carry_l1,
                    mlp_carry_l1,
                    embedding_carry_l1,
                    attention_carry_l1,
                    attention_q_carry_l1,
                    attention_k_carry_l1,
                    attention_v_carry_l1,
                    attention_o_carry_l1,
                    &adaptive_attention_shifts,
                    &model,
                ))?;
            }
            batch_start_index = batch_end_index;
        }
    }

    let final_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let final_total_error = final_eval.mistakes;
    let final_probability_error_q15 = final_eval.probability_error_q15;
    let final_mistakes = final_eval.mistakes;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = final_eval.logits_hash;
    if progress_interval_batches > 0 {
        progress(&mini_transformer_training_progress_trace(
            config,
            tokens.len(),
            token_hash,
            window_hash,
            starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            rejected_window_count,
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            output_head_carry_l1,
            mlp_carry_l1,
            embedding_carry_l1,
            attention_carry_l1,
            attention_q_carry_l1,
            attention_k_carry_l1,
            attention_v_carry_l1,
            attention_o_carry_l1,
            &adaptive_attention_shifts,
            &model,
        ))?;
    }

    let trace = MiniTransformerMlpTrainingTrace {
        trace_detail,
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        accepted_batch_count,
        rejected_batch_count,
        output_head_accumulator_batch_count,
        output_head_accumulator_window_count,
        mlp_accumulator_batch_count,
        mlp_accumulator_window_count,
        attention_accumulator_batch_count,
        attention_accumulator_window_count,
        embedding_accumulator_batch_count,
        embedding_accumulator_window_count,
        rollback_count,
        rejected_window_count,
        loss_regression_rejected_batch_count,
        final_invalid_forward_count: final_eval.invalid_forward_count,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        initial_embedding_hash,
        final_embedding_hash: model.embedding_hash(),
        initial_output_head_hash,
        final_output_head_hash: model.output_head_hash(),
        initial_mlp_hash,
        final_mlp_hash: model.mlp_hash(),
        initial_attention_hash,
        final_attention_hash: model.attention_hash(),
        initial_attention_q_hash,
        final_attention_q_hash: model.attention_q_hash(),
        initial_attention_k_hash,
        final_attention_k_hash: model.attention_k_hash(),
        initial_attention_v_hash,
        final_attention_v_hash: model.attention_v_hash(),
        initial_attention_o_hash,
        final_attention_o_hash: model.attention_o_hash(),
        initial_total_error,
        final_total_error,
        initial_probability_error_q15,
        final_probability_error_q15,
        initial_mistakes,
        final_mistakes,
        output_head_saturation_count,
        mlp_saturation_count,
        embedding_saturation_count,
        attention_saturation_count,
        residual_saturation_count,
        output_head_zero_delta_count,
        mlp_zero_delta_count,
        embedding_zero_delta_count,
        attention_zero_delta_count,
        output_head_delta_l1,
        mlp_delta_l1,
        embedding_delta_l1,
        attention_delta_l1,
        attention_q_delta_l1,
        attention_k_delta_l1,
        attention_v_delta_l1,
        attention_o_delta_l1,
        output_head_carry_l1,
        mlp_carry_l1,
        embedding_carry_l1,
        attention_carry_l1,
        attention_q_carry_l1,
        attention_k_carry_l1,
        attention_v_carry_l1,
        attention_o_carry_l1,
        adaptive_rule_shift_adjustment_count: adaptive_attention_shifts.rule_adjustment_count,
        adaptive_rule_update_count: adaptive_attention_shifts.rule_update_count,
        adaptive_rule_event_count: adaptive_attention_shifts.rule_event_count,
        adaptive_holographic_shift_adjustment_count: adaptive_attention_shifts
            .holographic_adjustment_count,
        adaptive_holographic_update_count: adaptive_attention_shifts.total_memory_updates(),
        adaptive_holographic_hash: adaptive_attention_shifts.memory_hash(),
        adaptive_attention_shift_adjustment_count: adaptive_attention_shifts.adjustment_count,
        adaptive_attention_holographic_update_count: adaptive_attention_shifts
            .attention_memory_updates(),
        adaptive_attention_holographic_hash: adaptive_attention_shifts.attention_memory_hash(),
        final_output_learning_rate_shift: adaptive_attention_shifts.output_learning_rate_shift,
        final_mlp_learning_rate_shift: adaptive_attention_shifts.mlp_learning_rate_shift,
        final_embedding_learning_rate_shift: adaptive_attention_shifts
            .embedding_learning_rate_shift,
        final_attention_learning_rate_shift: adaptive_attention_shifts
            .attention_learning_rate_shift,
        final_attention_q_learning_rate_shift: adaptive_attention_shifts
            .attention_q_learning_rate_shift,
        final_attention_qk_learning_rate_shift: adaptive_attention_shifts
            .attention_qk_learning_rate_shift,
        final_accuracy_per_mille,
        final_logits_hash,
        adaptive_shift_events,
        steps,
    };

    for event in &trace.adaptive_shift_events {
        binary_trace(MiniTransformerBinaryTraceRecord::AdaptiveShift(event))?;
    }
    binary_trace(MiniTransformerBinaryTraceRecord::FinalSummary(&trace))?;

    Ok(MiniTransformerMlpTrainingRun { trace, model })
}

fn validate_mini_transformer_batch_mode(
    config: MiniTransformerMlpTrainConfig,
) -> Result<(), TrainError> {
    match config.batch_mode {
        MiniTransformerBatchMode::Serial => Ok(()),
        MiniTransformerBatchMode::MapReduce => {
            if config.batch_windows <= 1
                || config.attention_vo_error_feedback
                || config.attention_vo_oracle
                || config.reject_loss_regression
                || config.attention_kind.uses_incremental_state()
            {
                return Err(TrainError::InvalidConfig);
            }

            Ok(())
        }
    }
}

fn mini_transformer_effective_map_reduce_workers(config: MiniTransformerMlpTrainConfig) -> usize {
    if config.batch_mode != MiniTransformerBatchMode::MapReduce {
        return 1;
    }
    if config.map_reduce_workers == 0 {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
            .max(1)
    } else {
        config.map_reduce_workers.max(1)
    }
}

struct MiniTransformerMapReduceBatchResult {
    accepted_window_count: usize,
    output_head_gradient: LinearWeightGradientI64,
    mlp_weight_gradients: Vec<GatedMlpWeightGradientI64>,
    attention_weight_gradients: Vec<MiniTransformerAttentionWeightGradientI64>,
    rms_weight_gradients: Vec<MiniTransformerRmsWeightGradientI64>,
    embedding_gradient: MiniTransformerEmbeddingGradientI64,
    mlp_saturation_count: usize,
    attention_saturation_count: usize,
    residual_saturation_count: usize,
    steps: Vec<MiniTransformerMlpTrainingStepTrace>,
}

impl MiniTransformerMapReduceBatchResult {
    fn new(
        config: MiniTransformerMlpTrainConfig,
        transformer_layers: usize,
    ) -> Result<Self, TrainError> {
        Ok(Self {
            accepted_window_count: 0,
            output_head_gradient: LinearWeightGradientI64::new(
                MINI_TRANSFORMER_D_MODEL,
                BYTE_VOCAB,
            )
            .ok_or(TrainError::InvalidConfig)?,
            mlp_weight_gradients: mini_transformer_new_gated_mlp_weight_gradients(
                transformer_layers,
            )?,
            attention_weight_gradients: mini_transformer_new_attention_weight_gradients(
                transformer_layers,
            )?,
            rms_weight_gradients: (0..transformer_layers)
                .map(|_| MiniTransformerRmsWeightGradientI64::new())
                .collect(),
            embedding_gradient: MiniTransformerEmbeddingGradientI64::new(config.seq_len)
                .ok_or(TrainError::InvalidConfig)?,
            mlp_saturation_count: 0,
            attention_saturation_count: 0,
            residual_saturation_count: 0,
            steps: Vec::new(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_map_reduce_batch(
    tokens: &[u8],
    starts: &[usize],
    target_frequency_weights_q15: &[i16; BYTE_VOCAB],
    batch_start_index: usize,
    batch_end_index: usize,
    epoch: usize,
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
    updates_before_batch: usize,
    trace_detail: MiniTransformerTraceDetail,
    trace_sample_interval: usize,
) -> Result<MiniTransformerMapReduceBatchResult, TrainError> {
    if batch_start_index >= batch_end_index
        || batch_end_index > starts.len()
        || config.batch_mode != MiniTransformerBatchMode::MapReduce
    {
        return Err(TrainError::InvalidConfig);
    }

    let batch_len = batch_end_index - batch_start_index;
    let worker_count = mini_transformer_effective_map_reduce_workers(config)
        .min(batch_len)
        .max(1);
    if worker_count == 1 {
        return mini_transformer_map_reduce_worker_batch(
            tokens,
            starts,
            target_frequency_weights_q15,
            batch_start_index,
            batch_end_index,
            batch_start_index,
            epoch,
            model,
            config,
            updates_before_batch,
            trace_detail,
            trace_sample_interval,
        );
    }

    let chunk_size = batch_len.div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut chunk_start = batch_start_index;
        while chunk_start < batch_end_index {
            let start = chunk_start;
            let end = start.saturating_add(chunk_size).min(batch_end_index);
            handles.push(scope.spawn(move || {
                mini_transformer_map_reduce_worker_batch(
                    tokens,
                    starts,
                    target_frequency_weights_q15,
                    start,
                    end,
                    batch_start_index,
                    epoch,
                    model,
                    config,
                    updates_before_batch,
                    trace_detail,
                    trace_sample_interval,
                )
            }));
            chunk_start = end;
        }

        let mut result =
            MiniTransformerMapReduceBatchResult::new(config, model.transformer_layers())?;
        for handle in handles {
            let worker = match handle.join() {
                Ok(worker) => worker?,
                Err(payload) => std::panic::resume_unwind(payload),
            };
            mini_transformer_merge_map_reduce_batch_result(&mut result, worker)?;
        }
        Ok(result)
    })
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_map_reduce_worker_batch(
    tokens: &[u8],
    starts: &[usize],
    target_frequency_weights_q15: &[i16; BYTE_VOCAB],
    range_start_index: usize,
    range_end_index: usize,
    batch_start_index: usize,
    epoch: usize,
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
    updates_before_batch: usize,
    trace_detail: MiniTransformerTraceDetail,
    trace_sample_interval: usize,
) -> Result<MiniTransformerMapReduceBatchResult, TrainError> {
    if range_start_index > range_end_index || range_end_index > starts.len() || config.seq_len == 0
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut result = MiniTransformerMapReduceBatchResult::new(config, model.transformer_layers())?;
    let mut model_for_backward = model.clone();
    let last_start = (config.seq_len - 1)
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let last_end = last_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let mut workspace = MiniTransformerHostTrainCoreWorkspaceBuffers::new(config.seq_len)?;
    workspace.validate_host_training_step_shape(config.seq_len)?;

    for (window_index, &window_start) in starts
        .iter()
        .enumerate()
        .take(range_end_index)
        .skip(range_start_index)
    {
        let context_end = window_start
            .checked_add(config.seq_len)
            .ok_or(TrainError::InvalidConfig)?;
        if context_end >= tokens.len() {
            return Err(TrainError::InvalidConfig);
        }
        let context = &tokens[window_start..context_end];
        let target_token = tokens[context_end];
        let cache_before = mini_transformer_forward_for_attention_and_position(
            model,
            context,
            config.attention_kind,
            config.position_policy,
        )
        .map_err(|_| TrainError::CoreRejected("mini_transformer_map_reduce_forward"))?;
        let mut gradient_q15 =
            byte_vocab_softmax_gradient_q15(&cache_before.probabilities_q15, target_token);
        apply_byte_argmax_margin_gradient_q15(
            &mut gradient_q15,
            &cache_before.logits_q8,
            target_token,
            config.argmax_margin_weight_q15,
        );
        let target_frequency_weight_q15 = target_frequency_weights_q15[usize::from(target_token)];
        let weighted_gradient_q15 =
            byte_scale_gradient_q15(&gradient_q15, target_frequency_weight_q15);
        let grad_output_q15 = byte_gradient_i32_to_i16(&weighted_gradient_q15);
        workspace.reset_host_training_step();
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_output_q15,
            LinearBackwardInputI16I8Params {
                weights: &model.output_weights,
                forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: BYTE_VOCAB,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.output_scaled_grad,
            },
            &mut workspace.grad_last_features,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_output_head_backward_input",
        ))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &cache_before.output_features,
            &workspace.output_scaled_grad,
            &mut result.output_head_gradient,
        )?;

        if cache_before.layers.len() != result.mlp_weight_gradients.len()
            || cache_before.layers.len() != result.attention_weight_gradients.len()
            || cache_before.layers.len() != result.rms_weight_gradients.len()
        {
            return Err(TrainError::InvalidConfig);
        }
        let (
            mlp_input_saturation_count,
            attention_gradient_saturation_count,
            gradient_residual_saturation_count,
            embedding_gradient_saturation_count,
        ) = if cache_before.layers.len() > 1 {
            let total = config
                .seq_len
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?;
            let mut grad_block_output = vec![0_i16; total];
            grad_block_output[last_start..last_end].copy_from_slice(&workspace.grad_last_features);
            let mut stacked_mlp_input_saturation_count = 0_usize;
            let mut stacked_attention_gradient_saturation_count = 0_usize;
            let mut stacked_gradient_residual_saturation_count = 0_usize;
            let mut stacked_input_gradient_saturation_count = 0_usize;

            for layer_index in (0..cache_before.layers.len()).rev() {
                let layer_runtime_config = mini_transformer_stacked_layer_runtime_config(
                    config,
                    layer_index,
                    cache_before.layers.len(),
                );
                let block_accumulation = mini_transformer_block_backward_accumulate_i64_checked(
                    &cache_before.layers[layer_index],
                    &grad_block_output,
                    &mut model_for_backward,
                    layer_index,
                    layer_runtime_config,
                    &mut workspace,
                    &mut result.mlp_weight_gradients[layer_index],
                    &mut result.attention_weight_gradients[layer_index],
                    &mut result.rms_weight_gradients[layer_index],
                )?;
                stacked_mlp_input_saturation_count = stacked_mlp_input_saturation_count
                    .saturating_add(block_accumulation.mlp_input_saturation_count);
                stacked_attention_gradient_saturation_count =
                    stacked_attention_gradient_saturation_count
                        .saturating_add(block_accumulation.attention_gradient_saturation_count);
                stacked_gradient_residual_saturation_count =
                    stacked_gradient_residual_saturation_count
                        .saturating_add(block_accumulation.gradient_residual_saturation_count);
                stacked_input_gradient_saturation_count = stacked_input_gradient_saturation_count
                    .saturating_add(block_accumulation.input_gradient_saturation_count);
                grad_block_output = block_accumulation.grad_input;
            }

            workspace.grad_embedding_output[..total].copy_from_slice(&grad_block_output);
            accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                context,
                &workspace.grad_embedding_output,
                config.position_policy,
                &mut result.embedding_gradient,
            )?;
            (
                stacked_mlp_input_saturation_count,
                stacked_attention_gradient_saturation_count,
                stacked_gradient_residual_saturation_count,
                stacked_input_gradient_saturation_count,
            )
        } else {
            let block_cache = cache_before
                .layers
                .last()
                .ok_or(TrainError::InvalidConfig)?;
            let rms_weights = if model.rms_norm_enabled() {
                let range = model.rms_weight_range(0)?;
                Some((
                    model.attention_rms_weights[range.clone()].to_vec(),
                    model.mlp_rms_weights[range].to_vec(),
                ))
            } else {
                None
            };
            workspace.grad_mlp_output[last_start..last_end]
                .copy_from_slice(&workspace.grad_last_features);
            let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
                &workspace.grad_mlp_output,
                mini_transformer_final_mlp_params(model, config.seq_len)?,
                &cache_before.mlp_up,
                &cache_before.mlp_gate,
                GatedMlpBackwardScales {
                    down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                    up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                    gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                },
                GatedMlpBackwardWorkspace {
                    scaled_grad_output: &mut workspace.mlp_scaled_grad,
                    grad_gated: &mut workspace.mlp_input_grad_gated,
                    grad_up: &mut workspace.mlp_input_grad_up,
                    grad_gate: &mut workspace.mlp_input_grad_gate,
                    grad_up_input: &mut workspace.mlp_input_grad_up_input,
                    grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
                },
                &mut workspace.grad_mlp_input,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_mlp_backward_input",
            ))?;

            let mut grad_mlp_residual = vec![0_i16; workspace.grad_mlp_input.len()];
            let mlp_rms_saturation = if let Some((_, mlp_weights)) = &rms_weights {
                mini_transformer_rms_norm_backward_rows(
                    &block_cache.attention_residual,
                    mlp_weights,
                    &workspace.grad_mlp_input,
                    &mut grad_mlp_residual,
                    &mut result.rms_weight_gradients[0].mlp,
                )?
            } else {
                grad_mlp_residual.copy_from_slice(&workspace.grad_mlp_input);
                0
            };
            let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                &workspace.grad_mlp_output,
                &grad_mlp_residual,
                &mut workspace.grad_attention_output,
            )?;

            accumulate_gated_mlp_weight_gradient_i64(
                &cache_before.mlp_norm,
                &workspace.grad_mlp_output,
                &cache_before.mlp_gated,
                &workspace.mlp_input_grad_up,
                &workspace.mlp_input_grad_gate,
                GatedMlpWeightUpdateParams {
                    up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                    seq_len: config.seq_len,
                    d_model: MINI_TRANSFORMER_D_MODEL,
                    hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.mlp_learning_rate_shift,
                },
                &mut result.mlp_weight_gradients[0],
                &mut workspace.mlp_scaled_grad,
            )?;

            let attention_update = mini_transformer_attention_update_i8_checked(
                block_cache,
                &mut model_for_backward,
                0,
                config,
                &mut workspace,
                Some(&mut result.attention_weight_gradients[0]),
            )?;
            let mut grad_attention_input = vec![0_i16; workspace.grad_attention_norm_input.len()];
            let attention_rms_saturation = if let Some((attention_weights, _)) = &rms_weights {
                mini_transformer_rms_norm_backward_rows(
                    &block_cache.block_input,
                    attention_weights,
                    &workspace.grad_attention_norm_input,
                    &mut grad_attention_input,
                    &mut result.rms_weight_gradients[0].attention,
                )?
            } else {
                grad_attention_input.copy_from_slice(&workspace.grad_attention_norm_input);
                0
            };
            let embedding_gradient_saturation_count = add_i16_residual_rows_checked(
                &workspace.grad_attention_output,
                &grad_attention_input,
                &mut workspace.grad_embedding_output,
            )?;
            accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                context,
                &workspace.grad_embedding_output,
                config.position_policy,
                &mut result.embedding_gradient,
            )?;
            (
                mlp_input_saturation_count.saturating_add(mlp_rms_saturation),
                attention_update.gradient_saturation_count,
                gradient_residual_saturation_count,
                embedding_gradient_saturation_count.saturating_add(attention_rms_saturation),
            )
        };

        result.accepted_window_count = result.accepted_window_count.saturating_add(1);
        result.mlp_saturation_count = result
            .mlp_saturation_count
            .saturating_add(mlp_input_saturation_count);
        result.attention_saturation_count = result
            .attention_saturation_count
            .saturating_add(attention_gradient_saturation_count);
        result.residual_saturation_count = result
            .residual_saturation_count
            .saturating_add(gradient_residual_saturation_count)
            .saturating_add(embedding_gradient_saturation_count)
            .saturating_add(cache_before.residual_saturation_count)
            .saturating_add(cache_before.residual_saturation_count);

        let update_index = updates_before_batch
            .saturating_add(window_index.saturating_sub(batch_start_index))
            .saturating_add(1);
        if mini_transformer_should_record_step(trace_detail, update_index, trace_sample_interval) {
            let predicted_token_before = byte_argmax_i32(&cache_before.logits_q8);
            let output_head_hash = model.output_head_hash();
            let mlp_hash = model.mlp_hash();
            let attention_hash = model.attention_hash();
            let embedding_hash = model.embedding_hash();
            result.steps.push(MiniTransformerMlpTrainingStepTrace {
                update_index,
                epoch,
                window_index,
                window_start,
                first_token: tokens[window_start],
                last_token: tokens[window_start + config.seq_len - 1],
                target_token,
                predicted_token_before,
                predicted_token_after: predicted_token_before,
                target_probability_before_q15: cache_before.probabilities_q15
                    [usize::from(target_token)],
                target_probability_after_q15: cache_before.probabilities_q15
                    [usize::from(target_token)],
                embedding_cache_hash: hash_i16_slice(&cache_before.embedding_output),
                attention_cache_hash: hash_i16_slice(&cache_before.attention_output),
                mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                block_output_hash_before: hash_i16_slice(&cache_before.block_output),
                block_output_hash_after: hash_i16_slice(&cache_before.block_output),
                output_head_hash_before: output_head_hash,
                output_head_hash_after: output_head_hash,
                mlp_hash_before: mlp_hash,
                mlp_hash_after: mlp_hash,
                attention_hash_before: attention_hash,
                attention_hash_after: attention_hash,
                embedding_hash_before: embedding_hash,
                embedding_hash_after: embedding_hash,
                output_head_saturation_count: 0,
                mlp_saturation_count: mlp_input_saturation_count,
                embedding_saturation_count: 0,
                attention_saturation_count: attention_gradient_saturation_count,
                residual_saturation_count: gradient_residual_saturation_count
                    + embedding_gradient_saturation_count
                    + cache_before.residual_saturation_count
                    + cache_before.residual_saturation_count,
                output_head_zero_delta_count: 0,
                mlp_zero_delta_count: 0,
                embedding_zero_delta_count: 0,
                attention_zero_delta_count: 0,
                output_head_delta_l1: 0,
                mlp_delta_l1: 0,
                embedding_delta_l1: 0,
                attention_delta_l1: 0,
                attention_q_delta_l1: 0,
                attention_k_delta_l1: 0,
                attention_v_delta_l1: 0,
                attention_o_delta_l1: 0,
            });
        }
    }

    Ok(result)
}

fn mini_transformer_merge_map_reduce_batch_result(
    target: &mut MiniTransformerMapReduceBatchResult,
    source: MiniTransformerMapReduceBatchResult,
) -> Result<(), TrainError> {
    target.accepted_window_count = target
        .accepted_window_count
        .checked_add(source.accepted_window_count)
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_window_count",
        ))?;
    mini_transformer_merge_linear_weight_gradient_i64(
        &mut target.output_head_gradient,
        &source.output_head_gradient,
    )?;
    mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
        &mut target.mlp_weight_gradients,
        &source.mlp_weight_gradients,
    )?;
    mini_transformer_merge_attention_weight_gradient_i64_layers(
        &mut target.attention_weight_gradients,
        &source.attention_weight_gradients,
    )?;
    mini_transformer_merge_rms_weight_gradient_i64_layers(
        &mut target.rms_weight_gradients,
        &source.rms_weight_gradients,
    )?;
    mini_transformer_merge_embedding_gradient_i64(
        &mut target.embedding_gradient,
        &source.embedding_gradient,
    )?;
    target.mlp_saturation_count = target
        .mlp_saturation_count
        .saturating_add(source.mlp_saturation_count);
    target.attention_saturation_count = target
        .attention_saturation_count
        .saturating_add(source.attention_saturation_count);
    target.residual_saturation_count = target
        .residual_saturation_count
        .saturating_add(source.residual_saturation_count);
    target.steps.extend(source.steps);
    Ok(())
}

fn mini_transformer_merge_linear_weight_gradient_i64(
    target: &mut LinearWeightGradientI64,
    source: &LinearWeightGradientI64,
) -> Result<(), TrainError> {
    if target.input_dim != source.input_dim
        || target.output_dim != source.output_dim
        || target.accumulators.len() != source.accumulators.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count =
        target
            .sample_count
            .checked_add(source.sample_count)
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_sample_count",
            ))?;
    for (target, source) in target
        .accumulators
        .iter_mut()
        .zip(source.accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_accumulator",
        ))?;
    }
    Ok(())
}

fn mini_transformer_merge_rms_vector_gradient_i64(
    target: &mut MiniTransformerRmsVectorGradientI64,
    source: &MiniTransformerRmsVectorGradientI64,
) -> Result<(), TrainError> {
    if target.accumulators.len() != source.accumulators.len() {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count = target
        .sample_count
        .checked_add(source.sample_count)
        .ok_or(TrainError::CoreRejected("RMSNorm sample count overflow"))?;
    for (target, source) in target
        .accumulators
        .iter_mut()
        .zip(source.accumulators.iter())
    {
        *target = target
            .checked_add(*source)
            .ok_or(TrainError::CoreRejected("RMSNorm gradient overflow"))?;
    }
    Ok(())
}

fn mini_transformer_merge_rms_weight_gradient_i64_layers(
    target: &mut [MiniTransformerRmsWeightGradientI64],
    source: &[MiniTransformerRmsWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_rms_vector_gradient_i64(&mut target.attention, &source.attention)?;
        mini_transformer_merge_rms_vector_gradient_i64(&mut target.mlp, &source.mlp)?;
    }
    Ok(())
}

fn mini_transformer_merge_gated_mlp_weight_gradient_i64(
    target: &mut GatedMlpWeightGradientI64,
    source: &GatedMlpWeightGradientI64,
) -> Result<(), TrainError> {
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.down, &source.down)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.up, &source.up)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.gate, &source.gate)?;
    Ok(())
}

fn mini_transformer_new_gated_mlp_weight_gradients(
    layer_count: usize,
) -> Result<Vec<GatedMlpWeightGradientI64>, TrainError> {
    if layer_count == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let mut gradients = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        gradients.push(
            GatedMlpWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HIDDEN_DIM)
                .ok_or(TrainError::InvalidConfig)?,
        );
    }
    Ok(gradients)
}

fn mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
    gradients: &mut [GatedMlpWeightGradientI64],
) {
    for gradient in gradients {
        gradient.clear();
    }
}

fn mini_transformer_gated_mlp_weight_gradient_i64_layers_residual_l1(
    gradients: &[GatedMlpWeightGradientI64],
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        total.saturating_add(gradient.residual_l1())
    })
}

fn mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
    target: &mut [GatedMlpWeightGradientI64],
    source: &[GatedMlpWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() || target.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_gated_mlp_weight_gradient_i64(target, source)?;
    }
    Ok(())
}

fn mini_transformer_merge_attention_weight_gradient_i64(
    target: &mut MiniTransformerAttentionWeightGradientI64,
    source: &MiniTransformerAttentionWeightGradientI64,
) -> Result<(), TrainError> {
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.q, &source.q)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.k, &source.k)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.v, &source.v)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.o, &source.o)?;
    Ok(())
}

fn mini_transformer_new_attention_weight_gradients(
    layer_count: usize,
) -> Result<Vec<MiniTransformerAttentionWeightGradientI64>, TrainError> {
    if layer_count == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let mut gradients = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        gradients.push(
            MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?,
        );
    }
    Ok(gradients)
}

fn mini_transformer_clear_attention_weight_gradient_i64_layers(
    gradients: &mut [MiniTransformerAttentionWeightGradientI64],
) {
    for gradient in gradients {
        gradient.clear();
    }
}

fn mini_transformer_attention_weight_gradient_i64_layers_residual_l1(
    gradients: &[MiniTransformerAttentionWeightGradientI64],
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        total.saturating_add(gradient.residual_l1())
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MiniTransformerAttentionProjection {
    Query,
    Key,
    Value,
    Output,
}

fn mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
    gradients: &[MiniTransformerAttentionWeightGradientI64],
    projection: MiniTransformerAttentionProjection,
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        let projection_l1 = match projection {
            MiniTransformerAttentionProjection::Query => gradient.q.residual_l1(),
            MiniTransformerAttentionProjection::Key => gradient.k.residual_l1(),
            MiniTransformerAttentionProjection::Value => gradient.v.residual_l1(),
            MiniTransformerAttentionProjection::Output => gradient.o.residual_l1(),
        };
        total.saturating_add(projection_l1)
    })
}

fn mini_transformer_merge_attention_weight_gradient_i64_layers(
    target: &mut [MiniTransformerAttentionWeightGradientI64],
    source: &[MiniTransformerAttentionWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() || target.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_attention_weight_gradient_i64(target, source)?;
    }
    Ok(())
}

fn mini_transformer_merge_embedding_gradient_i64(
    target: &mut MiniTransformerEmbeddingGradientI64,
    source: &MiniTransformerEmbeddingGradientI64,
) -> Result<(), TrainError> {
    if target.token_accumulators.len() != source.token_accumulators.len()
        || target.position_accumulators.len() != source.position_accumulators.len()
        || target.token_residuals.len() != source.token_residuals.len()
        || target.position_residuals.len() != source.position_residuals.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count =
        target
            .sample_count
            .checked_add(source.sample_count)
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_embedding_sample_count",
            ))?;
    for (target, source) in target
        .token_accumulators
        .iter_mut()
        .zip(source.token_accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_embedding_accumulator",
        ))?;
    }
    for (target, source) in target
        .position_accumulators
        .iter_mut()
        .zip(source.position_accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_position_accumulator",
        ))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerBinaryTraceRecord<'a> {
    Header { initial_model_hash: u64 },
    StepSample(&'a MiniTransformerMlpTrainingStepTrace),
    AdaptiveShift(&'a MiniTransformerAdaptiveShiftEventTrace),
    FinalSummary(&'a MiniTransformerMlpTrainingTrace),
}

pub struct MiniTransformerBinaryTraceWriter<W: std::io::Write> {
    writer: W,
}

impl<W: std::io::Write> MiniTransformerBinaryTraceWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write_record(
        &mut self,
        record: MiniTransformerBinaryTraceRecord<'_>,
    ) -> std::io::Result<()> {
        match record {
            MiniTransformerBinaryTraceRecord::Header { initial_model_hash } => self
                .writer
                .write_all(&mini_transformer_binary_trace_header_v1(initial_model_hash)),
            MiniTransformerBinaryTraceRecord::StepSample(step) => self
                .writer
                .write_all(&mini_transformer_binary_step_sample_record_v1(step)),
            MiniTransformerBinaryTraceRecord::AdaptiveShift(event) => self
                .writer
                .write_all(&mini_transformer_binary_adaptive_shift_record_v1(event)),
            MiniTransformerBinaryTraceRecord::FinalSummary(trace) => self
                .writer
                .write_all(&mini_transformer_binary_final_summary_record_v1(trace)),
        }
    }

    pub fn write_trace_tail(
        &mut self,
        trace: &MiniTransformerMlpTrainingTrace,
    ) -> std::io::Result<()> {
        for event in &trace.adaptive_shift_events {
            self.write_record(MiniTransformerBinaryTraceRecord::AdaptiveShift(event))?;
        }
        self.write_record(MiniTransformerBinaryTraceRecord::FinalSummary(trace))?;
        self.writer.flush()
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

pub fn mini_transformer_binary_trace_header_v1(initial_model_hash: u64) -> [u8; 16] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN);
    out.extend_from_slice(MINI_TRANSFORMER_BINARY_TRACE_MAGIC);
    out.push(MINI_TRANSFORMER_BINARY_TRACE_VERSION);
    out.push(MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID);
    push_binary_u16(&mut out, 0);
    push_binary_u64(&mut out, initial_model_hash);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_step_sample_record_v1(
    step: &MiniTransformerMlpTrainingStepTrace,
) -> [u8; 32] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN);
    push_mini_transformer_binary_step_sample(&mut out, step);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_adaptive_shift_record_v1(
    event: &MiniTransformerAdaptiveShiftEventTrace,
) -> [u8; 22] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN);
    push_mini_transformer_binary_adaptive_shift(&mut out, event);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_final_summary_record_v1(
    trace: &MiniTransformerMlpTrainingTrace,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN);
    push_mini_transformer_binary_final_summary(&mut out, trace);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN);
    out
}

impl MiniTransformerMlpSwarmTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", "wiki_bard_mini_transformer_mlp_swarm");
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trace_detail",
            self.swarm_config.trace_detail.as_str(),
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_model_hash", self.final_model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_workers_field(&mut out, "workers", &self.workers);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmTrainingProgressTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_PROGRESS_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_progress",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trace_detail",
            self.swarm_config.trace_detail.as_str(),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_progress_workers_field(&mut out, "workers", &self.workers);
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmScalingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_SCALING_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_scaling",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"host\":{");
        push_usize_field(
            &mut out,
            "available_parallelism",
            self.available_parallelism,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"benchmark\":{");
        push_usize_field(
            &mut out,
            "requested_max_workers",
            self.requested_max_workers,
        );
        comma(&mut out);
        push_usize_field(&mut out, "run_count", self.runs.len());
        comma(&mut out);
        push_usize_array_field(&mut out, "worker_counts", &self.worker_counts);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_scaling_runs_field(&mut out, "runs", &self.runs);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_SWARM_SCALING_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

fn push_usize_array_field(out: &mut String, field: &str, values: &[usize]) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_mini_transformer_swarm_scaling_runs_field(
    out: &mut String,
    field: &str,
    runs: &[MiniTransformerMlpSwarmScalingRunTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, run) in runs.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_scaling_run(out, run);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_scaling_run(
    out: &mut String,
    run: &MiniTransformerMlpSwarmScalingRunTrace,
) {
    out.push('{');
    push_usize_field(out, "requested_worker_count", run.requested_worker_count);
    comma(out);
    push_usize_field(out, "effective_worker_count", run.effective_worker_count);
    comma(out);
    push_u64_field(out, "elapsed_ns", run.elapsed_ns);
    comma(out);
    push_milli_decimal_field(out, "elapsed_ms", run.elapsed_ns / 1_000);
    comma(out);
    push_u64_field(out, "speedup_per_mille", run.speedup_per_mille);
    comma(out);
    push_milli_decimal_field(out, "speedup", run.speedup_per_mille);
    comma(out);
    push_u64_field(
        out,
        "parallel_efficiency_per_mille",
        run.parallel_efficiency_per_mille,
    );
    comma(out);
    push_milli_decimal_field(
        out,
        "parallel_efficiency",
        run.parallel_efficiency_per_mille,
    );
    comma(out);
    push_u64_field(
        out,
        "windows_per_second_milli",
        run.windows_per_second_milli,
    );
    comma(out);
    push_milli_decimal_field(out, "windows_per_second", run.windows_per_second_milli);
    comma(out);
    push_u64_field(
        out,
        "updates_per_second_milli",
        run.updates_per_second_milli,
    );
    comma(out);
    push_milli_decimal_field(out, "updates_per_second", run.updates_per_second_milli);
    comma(out);
    push_usize_field(out, "examined_windows", run.examined_windows);
    comma(out);
    push_usize_field(out, "updates", run.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", run.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", run.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", run.rollback_count);
    comma(out);
    push_usize_field(out, "best_worker_index", run.best_worker_index);
    comma(out);
    push_usize_field(out, "best_final_total_error", run.best_final_total_error);
    comma(out);
    push_usize_field(
        out,
        "best_final_probability_error_q15",
        run.best_final_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "best_final_accuracy_per_mille",
        run.best_final_accuracy_per_mille,
    );
    comma(out);
    push_hash_field(out, "final_model_hash", run.final_model_hash);
    out.push('}');
}

fn push_mini_transformer_swarm_route_config_field(
    out: &mut String,
    field: &str,
    config: &MiniTransformerSwarmRouteConfig,
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('{');
    match config.required_capabilities.first().map(String::as_str) {
        Some(capability) => push_string_field(out, "required_capability", capability),
        None => {
            push_quoted(out, "required_capability");
            out.push_str(":null");
        }
    }
    comma(out);
    push_string_vec_field(out, "required_capabilities", &config.required_capabilities);
    comma(out);
    push_optional_usize_field(out, "max_artifact_bytes", config.max_artifact_bytes);
    comma(out);
    push_optional_usize_field(out, "max_parameter_bytes", config.max_parameter_bytes);
    comma(out);
    push_usize_field(out, "active_expert_limit", config.active_expert_limit);
    comma(out);
    push_bool_field(out, "prompt_affinity", config.prompt_affinity);
    comma(out);
    push_usize_field(
        out,
        "prompt_affinity_max_windows",
        config.prompt_affinity_max_windows,
    );
    out.push('}');
}

fn push_mini_transformer_swarm_route_candidates_field(
    out: &mut String,
    field: &str,
    candidates: &[MiniTransformerSwarmRouteCandidateTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_route_candidate(out, candidate);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_route_candidate(
    out: &mut String,
    candidate: &MiniTransformerSwarmRouteCandidateTrace,
) {
    out.push('{');
    push_usize_field(out, "expert_index", candidate.expert_index);
    comma(out);
    push_string_field(out, "expert_id", &candidate.expert_id);
    comma(out);
    push_bool_field(out, "accepted", candidate.accepted);
    comma(out);
    push_string_field(out, "reject_reason", candidate.reject_reason);
    comma(out);
    push_i64_field(out, "score", candidate.score);
    comma(out);
    push_i64_field(out, "manifest_score", candidate.manifest_score);
    comma(out);
    push_i64_field(
        out,
        "prompt_affinity_score",
        candidate.prompt_affinity_score,
    );
    comma(out);
    push_usize_field(out, "prompt_eval_windows", candidate.prompt_eval_windows);
    comma(out);
    push_optional_usize_field(
        out,
        "prompt_probability_error_q15",
        candidate.prompt_probability_error_q15,
    );
    comma(out);
    push_bool_field(out, "capability_match", candidate.capability_match);
    comma(out);
    push_string_vec_field(out, "matched_capabilities", &candidate.matched_capabilities);
    comma(out);
    push_string_vec_field(out, "missing_capabilities", &candidate.missing_capabilities);
    comma(out);
    push_hash_field(out, "model_hash", candidate.model_hash);
    comma(out);
    push_usize_field(out, "artifact_bytes", candidate.artifact_bytes);
    comma(out);
    push_usize_field(out, "parameter_bytes", candidate.parameter_bytes);
    comma(out);
    push_usize_field(out, "worker_count", candidate.worker_count);
    comma(out);
    push_usize_field(out, "context_seq_len", candidate.context_seq_len);
    comma(out);
    push_string_field(out, "default_composition", candidate.default_composition);
    out.push('}');
}

fn push_mini_transformer_swarm_workers_field(
    out: &mut String,
    field: &str,
    workers: &[MiniTransformerMlpSwarmWorkerTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, worker) in workers.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_worker(out, worker);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_worker(
    out: &mut String,
    worker: &MiniTransformerMlpSwarmWorkerTrace,
) {
    out.push('{');
    push_usize_field(out, "worker_index", worker.worker_index);
    comma(out);
    push_usize_field(out, "window_offset", worker.window_offset);
    comma(out);
    push_usize_field(out, "stride", worker.stride);
    comma(out);
    push_optional_usize_field(out, "max_windows", worker.max_windows);
    comma(out);
    push_hash_field(out, "token_hash", worker.token_hash);
    comma(out);
    push_hash_field(out, "window_hash", worker.window_hash);
    comma(out);
    push_usize_field(out, "windows", worker.windows);
    comma(out);
    push_usize_field(out, "examined_windows", worker.examined_windows);
    comma(out);
    push_usize_field(out, "updates", worker.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", worker.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", worker.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", worker.rollback_count);
    comma(out);
    push_usize_field(out, "rejected_window_count", worker.rejected_window_count);
    comma(out);
    push_usize_field(
        out,
        "final_invalid_forward_count",
        worker.final_invalid_forward_count,
    );
    comma(out);
    push_usize_field(out, "initial_total_error", worker.initial_total_error);
    comma(out);
    push_usize_field(out, "final_total_error", worker.final_total_error);
    comma(out);
    push_usize_field(
        out,
        "initial_probability_error_q15",
        worker.initial_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "final_probability_error_q15",
        worker.final_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "final_accuracy_per_mille",
        worker.final_accuracy_per_mille,
    );
    comma(out);
    push_hash_field(out, "final_model_hash", worker.final_model_hash);
    comma(out);
    push_hash_field(out, "final_logits_hash", worker.final_logits_hash);
    out.push('}');
}

fn push_mini_transformer_swarm_progress_workers_field(
    out: &mut String,
    field: &str,
    workers: &[MiniTransformerMlpSwarmWorkerProgressTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, worker) in workers.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_progress_worker(out, worker);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_progress_worker(
    out: &mut String,
    worker: &MiniTransformerMlpSwarmWorkerProgressTrace,
) {
    let progress = &worker.progress;
    out.push('{');
    push_usize_field(out, "worker_index", worker.worker_index);
    comma(out);
    push_usize_field(out, "window_offset", progress.config.window_offset);
    comma(out);
    push_usize_field(out, "stride", progress.config.stride);
    comma(out);
    push_optional_usize_field(out, "max_windows", progress.config.max_windows);
    comma(out);
    push_hash_field(out, "token_hash", progress.token_hash);
    comma(out);
    push_hash_field(out, "window_hash", progress.window_hash);
    comma(out);
    push_usize_field(out, "windows", progress.windows);
    comma(out);
    push_usize_field(out, "examined_windows", progress.examined_windows);
    comma(out);
    push_usize_field(out, "updates", progress.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", progress.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", progress.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", progress.rollback_count);
    comma(out);
    push_usize_field(out, "rejected_window_count", progress.rejected_window_count);
    comma(out);
    push_u64_field(out, "output_head_delta_l1", progress.output_head_delta_l1);
    comma(out);
    push_u64_field(out, "mlp_delta_l1", progress.mlp_delta_l1);
    comma(out);
    push_u64_field(out, "embedding_delta_l1", progress.embedding_delta_l1);
    comma(out);
    push_u64_field(out, "attention_delta_l1", progress.attention_delta_l1);
    comma(out);
    push_u64_field(out, "attention_q_delta_l1", progress.attention_q_delta_l1);
    comma(out);
    push_u64_field(out, "attention_k_delta_l1", progress.attention_k_delta_l1);
    comma(out);
    push_u64_field(out, "attention_v_delta_l1", progress.attention_v_delta_l1);
    comma(out);
    push_u64_field(out, "attention_o_delta_l1", progress.attention_o_delta_l1);
    comma(out);
    push_u64_field(out, "output_head_carry_l1", progress.output_head_carry_l1);
    comma(out);
    push_u64_field(out, "mlp_carry_l1", progress.mlp_carry_l1);
    comma(out);
    push_u64_field(out, "embedding_carry_l1", progress.embedding_carry_l1);
    comma(out);
    push_u64_field(out, "attention_carry_l1", progress.attention_carry_l1);
    comma(out);
    push_u64_field(out, "attention_q_carry_l1", progress.attention_q_carry_l1);
    comma(out);
    push_u64_field(out, "attention_k_carry_l1", progress.attention_k_carry_l1);
    comma(out);
    push_u64_field(out, "attention_v_carry_l1", progress.attention_v_carry_l1);
    comma(out);
    push_u64_field(out, "attention_o_carry_l1", progress.attention_o_carry_l1);
    comma(out);
    push_usize_field(
        out,
        "adaptive_rule_shift_adjustment_count",
        progress.adaptive_rule_shift_adjustment_count,
    );
    comma(out);
    push_usize_field(
        out,
        "adaptive_holographic_shift_adjustment_count",
        progress.adaptive_holographic_shift_adjustment_count,
    );
    comma(out);
    push_hash_field(out, "current_model_hash", progress.model_hash);
    out.push('}');
}

impl MiniTransformerMlpTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_MLP_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", MINI_TRANSFORMER_MLP_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        comma(&mut out);
        push_hash_field(&mut out, "window_hash", self.window_hash);
        comma(&mut out);
        push_usize_field(&mut out, "windows", self.windows);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trained_component",
            "embedding_i16_plus_output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8",
        );
        comma(&mut out);
        push_string_field(&mut out, "attention", "updates_q_k_v_o_i8");
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        let attention_backward = match self.config.attention_kind {
            MiniTransformerAttentionKind::Base2Softmax => "base2_softmax_jacobian",
            MiniTransformerAttentionKind::Linear => {
                "linear_numerator_straight_through_denominator_constant"
            }
            MiniTransformerAttentionKind::LinearStreamingNope => "unsupported_for_training",
            MiniTransformerAttentionKind::LinearStreamingTttNope => "unsupported_for_training",
        };
        push_string_field(&mut out, "attention_backward", attention_backward);
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "activation", "hard_silu_shift2_q15");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_rule_shifts",
            self.config.adaptive_rule_shifts,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_interval_batches",
            self.config.adaptive_rule_interval_batches,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_attention_shifts",
            self.config.adaptive_attention_shifts,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_holographic_shifts",
            self.config.adaptive_holographic_shifts,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "attention_vo_error_feedback",
            self.config.attention_vo_error_feedback,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "attention_vo_oracle",
            self.config.attention_vo_oracle,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "reject_loss_regression",
            self.config.reject_loss_regression,
        );
        comma(&mut out);
        push_string_field(&mut out, "batch_mode", self.config.batch_mode.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "map_reduce_workers",
            self.config.map_reduce_workers,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "effective_map_reduce_workers",
            mini_transformer_effective_map_reduce_workers(self.config),
        );
        comma(&mut out);
        push_string_field(&mut out, "trace_detail", self.trace_detail.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_token_min",
            usize::from(self.config.target_token_min),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_token_max",
            usize::from(self.config.target_token_max),
        );
        comma(&mut out);
        push_string_field(
            &mut out,
            "target_segment",
            self.config.target_segment.as_str(),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_frequency_cap",
            self.config.target_frequency_cap as usize,
        );
        comma(&mut out);
        push_i16_field(
            &mut out,
            "target_frequency_min_weight_q15",
            self.config.target_frequency_min_weight_q15,
        );
        comma(&mut out);
        push_i16_field(
            &mut out,
            "argmax_margin_weight_q15",
            self.config.argmax_margin_weight_q15,
        );
        comma(&mut out);
        push_string_field(&mut out, "batch_mode", self.config.batch_mode.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "map_reduce_workers",
            self.config.map_reduce_workers,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "effective_map_reduce_workers",
            mini_transformer_effective_map_reduce_workers(self.config),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "batch_average_shift",
            usize::from(
                mini_transformer_batch_learning_rate_shift(self.config.batch_windows).unwrap_or(0),
            ),
        );
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        comma(&mut out);
        push_string_field(&mut out, "trace_detail", self.trace_detail.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rollback_history_limit",
            MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(&mut out, "initial_total_error", self.initial_total_error);
        comma(&mut out);
        push_usize_field(&mut out, "final_total_error", self.final_total_error);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "initial_probability_error_q15",
            self.initial_probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_probability_error_q15",
            self.final_probability_error_q15,
        );
        comma(&mut out);
        push_i32_field(
            &mut out,
            "probability_error_delta_i32",
            self.final_probability_error_q15 as i32 - self.initial_probability_error_q15 as i32,
        );
        comma(&mut out);
        push_usize_field(&mut out, "initial_mistakes", self.initial_mistakes);
        comma(&mut out);
        push_usize_field(&mut out, "final_mistakes", self.final_mistakes);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_accuracy_per_mille",
            self.final_accuracy_per_mille,
        );
        comma(&mut out);
        push_usize_field(&mut out, "accepted_batch_count", self.accepted_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rejected_batch_count", self.rejected_batch_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_accumulator_batch_count",
            self.output_head_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_accumulator_window_count",
            self.output_head_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_accumulator_batch_count",
            self.mlp_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_accumulator_window_count",
            self.mlp_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_accumulator_batch_count",
            self.attention_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_accumulator_window_count",
            self.attention_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_accumulator_batch_count",
            self.embedding_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_accumulator_window_count",
            self.embedding_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "rollback_count", self.rollback_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rejected_window_count",
            self.rejected_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "loss_regression_rejected_batch_count",
            self.loss_regression_rejected_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_invalid_forward_count",
            self.final_invalid_forward_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_saturation_count",
            self.output_head_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "mlp_saturation_count", self.mlp_saturation_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_saturation_count",
            self.embedding_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_saturation_count",
            self.attention_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "residual_saturation_count",
            self.residual_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_zero_delta_count",
            self.output_head_zero_delta_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "mlp_zero_delta_count", self.mlp_zero_delta_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_zero_delta_count",
            self.embedding_zero_delta_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_zero_delta_count",
            self.attention_zero_delta_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "output_head_delta_l1", self.output_head_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_delta_l1", self.mlp_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_delta_l1", self.embedding_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_delta_l1", self.attention_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_delta_l1", self.attention_q_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_delta_l1", self.attention_k_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_delta_l1", self.attention_v_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_delta_l1", self.attention_o_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "output_head_carry_l1", self.output_head_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_carry_l1", self.mlp_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_carry_l1", self.embedding_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_carry_l1", self.attention_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_carry_l1", self.attention_q_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_carry_l1", self.attention_k_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_carry_l1", self.attention_v_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_carry_l1", self.attention_o_carry_l1);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_shift_adjustment_count",
            self.adaptive_rule_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_update_count",
            self.adaptive_rule_update_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_event_count",
            self.adaptive_rule_event_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_trace_event_limit",
            MINI_TRANSFORMER_ADAPTIVE_RULE_TRACE_EVENT_LIMIT,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_shift_adjustment_count",
            self.adaptive_holographic_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_update_count",
            self.adaptive_holographic_update_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_meta_dim",
            MINI_TRANSFORMER_HOLO_META_DIM,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_action_count",
            MINI_TRANSFORMER_HOLO_ACTION_COUNT,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_memory_update_shift",
            MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT as usize,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_query_shift",
            MINI_TRANSFORMER_HOLO_QUERY_SHIFT as usize,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "adaptive_holographic_hash",
            self.adaptive_holographic_hash,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_attention_shift_adjustment_count",
            self.adaptive_attention_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_attention_holographic_update_count",
            self.adaptive_attention_holographic_update_count,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "adaptive_attention_holographic_hash",
            self.adaptive_attention_holographic_hash,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_output_learning_rate_shift",
            usize::from(self.final_output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_mlp_learning_rate_shift",
            usize::from(self.final_mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_embedding_learning_rate_shift",
            usize::from(self.final_embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_learning_rate_shift",
            usize::from(self.final_attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_q_learning_rate_shift",
            usize::from(self.final_attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_qk_learning_rate_shift",
            usize::from(self.final_attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_model_hash", self.initial_model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_model_hash", self.final_model_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_embedding_hash",
            self.initial_embedding_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_embedding_hash", self.final_embedding_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_output_head_hash",
            self.initial_output_head_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_output_head_hash",
            self.final_output_head_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "initial_mlp_hash", self.initial_mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_mlp_hash", self.final_mlp_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_hash",
            self.initial_attention_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_attention_hash", self.final_attention_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_q_hash",
            self.initial_attention_q_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_q_hash",
            self.final_attention_q_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_k_hash",
            self.initial_attention_k_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_k_hash",
            self.final_attention_k_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_v_hash",
            self.initial_attention_v_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_v_hash",
            self.final_attention_v_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_o_hash",
            self.initial_attention_o_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_o_hash",
            self.final_attention_o_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_mini_transformer_adaptive_shift_events_field(
            &mut out,
            "adaptive_shift_events",
            &self.adaptive_shift_events,
        );
        comma(&mut out);
        push_mini_transformer_mlp_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }

    pub fn to_binary_trace_v1(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN
                + self
                    .steps
                    .len()
                    .saturating_mul(MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN)
                + self
                    .adaptive_shift_events
                    .len()
                    .saturating_mul(MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN)
                + MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN,
        );
        out.extend_from_slice(&mini_transformer_binary_trace_header_v1(
            self.initial_model_hash,
        ));

        for step in &self.steps {
            out.extend_from_slice(&mini_transformer_binary_step_sample_record_v1(step));
        }
        for event in &self.adaptive_shift_events {
            out.extend_from_slice(&mini_transformer_binary_adaptive_shift_record_v1(event));
        }
        out.extend_from_slice(&mini_transformer_binary_final_summary_record_v1(self));
        out
    }
}

fn push_mini_transformer_binary_step_sample(
    out: &mut Vec<u8>,
    step: &MiniTransformerMlpTrainingStepTrace,
) {
    let start = out.len();
    out.push(MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE);
    push_binary_u32_clamped(out, step.update_index);
    push_binary_u32_clamped(out, step.window_start);
    out.push(step.first_token);
    out.push(step.last_token);
    out.push(step.target_token);
    out.push(step.predicted_token_before);
    out.push(step.predicted_token_after);
    push_binary_i16(out, step.target_probability_before_q15);
    push_binary_i16(out, step.target_probability_after_q15);
    push_binary_u16_clamped(out, step.residual_saturation_count);
    push_binary_u16_clamped(
        out,
        step.output_head_saturation_count
            .saturating_add(step.mlp_saturation_count)
            .saturating_add(step.embedding_saturation_count)
            .saturating_add(step.attention_saturation_count),
    );
    push_binary_u16_clamped(
        out,
        step.output_head_zero_delta_count
            .saturating_add(step.mlp_zero_delta_count)
            .saturating_add(step.embedding_zero_delta_count)
            .saturating_add(step.attention_zero_delta_count),
    );
    push_binary_u32_saturating(out, step.attention_delta_l1);
    push_binary_u32_saturating(
        out,
        step.output_head_delta_l1
            .saturating_add(step.mlp_delta_l1)
            .saturating_add(step.embedding_delta_l1)
            .saturating_add(step.attention_delta_l1),
    );
    debug_assert_eq!(out.len() - start, 32);
}

fn push_mini_transformer_binary_adaptive_shift(
    out: &mut Vec<u8>,
    event: &MiniTransformerAdaptiveShiftEventTrace,
) {
    out.push(MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT);
    push_binary_u32_clamped(out, event.batch_index);
    out.push(mini_transformer_binary_component_code(event.component));
    out.push(mini_transformer_binary_reason_code(event.reason));
    out.push(event.previous_shift);
    out.push(event.next_shift);
    out.push(event.delta as u8);
    push_binary_u16_clamped(out, event.observation_batches);
    push_binary_u16_clamped(out, event.rejected_batches);
    push_binary_u16_clamped(out, event.saturation_count);
    push_binary_u16_clamped(out, event.zero_delta_count);
    push_binary_u32_saturating(out, event.weight_delta_l1);
}

fn push_mini_transformer_binary_final_summary(
    out: &mut Vec<u8>,
    trace: &MiniTransformerMlpTrainingTrace,
) {
    out.push(MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY);
    out.push(mini_transformer_binary_trace_detail_code(
        trace.trace_detail,
    ));
    out.push(mini_transformer_binary_tokenizer_code(
        trace.config.tokenizer_id,
    ));
    out.push(mini_transformer_binary_attention_code(
        trace.config.attention_kind,
    ));
    out.push(mini_transformer_binary_position_code(
        trace.config.position_policy,
    ));
    push_binary_u16(out, mini_transformer_binary_config_flags(trace.config));
    push_binary_u32_clamped(out, trace.config.epochs);
    push_binary_u32_clamped(out, trace.config.seq_len);
    push_binary_u32_clamped(out, trace.config.stride);
    push_binary_u32_clamped(out, trace.config.window_offset);
    push_binary_optional_u32(out, trace.config.max_windows);
    push_binary_u32_clamped(out, trace.config.batch_windows);
    push_binary_i32(out, trace.config.learning_rate);
    out.push(trace.config.output_learning_rate_shift);
    out.push(trace.config.mlp_learning_rate_shift);
    out.push(trace.config.embedding_learning_rate_shift);
    out.push(trace.config.attention_learning_rate_shift);
    out.push(trace.config.attention_q_learning_rate_shift);
    out.push(trace.config.attention_qk_learning_rate_shift);

    push_binary_u64(out, trace.token_count as u64);
    push_binary_u64(out, trace.token_hash);
    push_binary_u64(out, trace.window_hash);
    push_binary_u64(out, trace.windows as u64);
    push_binary_u64(out, trace.examined_windows as u64);
    push_binary_u64(out, trace.updates as u64);
    push_binary_u64(out, trace.accepted_batch_count as u64);
    push_binary_u64(out, trace.rejected_batch_count as u64);
    push_binary_u64(out, trace.rollback_count as u64);
    push_binary_u64(out, trace.rejected_window_count as u64);
    push_binary_u64(out, trace.loss_regression_rejected_batch_count as u64);
    push_binary_u64(out, trace.final_invalid_forward_count as u64);

    push_binary_u64(out, trace.initial_total_error as u64);
    push_binary_u64(out, trace.final_total_error as u64);
    push_binary_u64(out, trace.initial_probability_error_q15 as u64);
    push_binary_u64(out, trace.final_probability_error_q15 as u64);
    push_binary_i64(
        out,
        trace.final_probability_error_q15 as i64 - trace.initial_probability_error_q15 as i64,
    );
    push_binary_u64(out, trace.initial_mistakes as u64);
    push_binary_u64(out, trace.final_mistakes as u64);
    push_binary_u16_clamped(out, trace.final_accuracy_per_mille);

    push_binary_u64(out, trace.output_head_saturation_count as u64);
    push_binary_u64(out, trace.mlp_saturation_count as u64);
    push_binary_u64(out, trace.embedding_saturation_count as u64);
    push_binary_u64(out, trace.attention_saturation_count as u64);
    push_binary_u64(out, trace.residual_saturation_count as u64);
    push_binary_u64(out, trace.output_head_zero_delta_count as u64);
    push_binary_u64(out, trace.mlp_zero_delta_count as u64);
    push_binary_u64(out, trace.embedding_zero_delta_count as u64);
    push_binary_u64(out, trace.attention_zero_delta_count as u64);
    push_binary_u64(out, trace.output_head_delta_l1);
    push_binary_u64(out, trace.mlp_delta_l1);
    push_binary_u64(out, trace.embedding_delta_l1);
    push_binary_u64(out, trace.attention_delta_l1);
    push_binary_u64(out, trace.attention_q_delta_l1);
    push_binary_u64(out, trace.attention_k_delta_l1);
    push_binary_u64(out, trace.attention_v_delta_l1);
    push_binary_u64(out, trace.attention_o_delta_l1);

    push_binary_u64(out, trace.adaptive_rule_shift_adjustment_count as u64);
    push_binary_u64(out, trace.adaptive_rule_update_count as u64);
    push_binary_u64(out, trace.adaptive_rule_event_count as u64);
    push_binary_u64(
        out,
        trace.adaptive_holographic_shift_adjustment_count as u64,
    );
    push_binary_u64(out, trace.adaptive_holographic_update_count as u64);
    push_binary_u64(out, trace.adaptive_holographic_hash);
    push_binary_u64(out, trace.adaptive_attention_shift_adjustment_count as u64);
    push_binary_u64(
        out,
        trace.adaptive_attention_holographic_update_count as u64,
    );
    push_binary_u64(out, trace.adaptive_attention_holographic_hash);
    out.push(trace.final_output_learning_rate_shift);
    out.push(trace.final_mlp_learning_rate_shift);
    out.push(trace.final_embedding_learning_rate_shift);
    out.push(trace.final_attention_learning_rate_shift);
    out.push(trace.final_attention_q_learning_rate_shift);
    out.push(trace.final_attention_qk_learning_rate_shift);

    push_binary_u64(out, trace.initial_model_hash);
    push_binary_u64(out, trace.final_model_hash);
    push_binary_u64(out, trace.initial_embedding_hash);
    push_binary_u64(out, trace.final_embedding_hash);
    push_binary_u64(out, trace.initial_output_head_hash);
    push_binary_u64(out, trace.final_output_head_hash);
    push_binary_u64(out, trace.initial_mlp_hash);
    push_binary_u64(out, trace.final_mlp_hash);
    push_binary_u64(out, trace.initial_attention_hash);
    push_binary_u64(out, trace.final_attention_hash);
    push_binary_u64(out, trace.initial_attention_q_hash);
    push_binary_u64(out, trace.final_attention_q_hash);
    push_binary_u64(out, trace.initial_attention_k_hash);
    push_binary_u64(out, trace.final_attention_k_hash);
    push_binary_u64(out, trace.initial_attention_v_hash);
    push_binary_u64(out, trace.final_attention_v_hash);
    push_binary_u64(out, trace.initial_attention_o_hash);
    push_binary_u64(out, trace.final_attention_o_hash);
    push_binary_u64(out, trace.final_logits_hash);
}

fn mini_transformer_binary_trace_detail_code(trace_detail: MiniTransformerTraceDetail) -> u8 {
    match trace_detail {
        MiniTransformerTraceDetail::Full => 0,
        MiniTransformerTraceDetail::Summary => 1,
        MiniTransformerTraceDetail::None => 2,
    }
}

fn mini_transformer_binary_tokenizer_code(tokenizer: ByteTokenizerId) -> u8 {
    match tokenizer {
        ByteTokenizerId::Identity => 0,
        ByteTokenizerId::AsciiLowerText => 1,
    }
}

fn mini_transformer_binary_attention_code(attention: MiniTransformerAttentionKind) -> u8 {
    match attention {
        MiniTransformerAttentionKind::Base2Softmax => 0,
        MiniTransformerAttentionKind::Linear => 1,
        MiniTransformerAttentionKind::LinearStreamingNope => 2,
        MiniTransformerAttentionKind::LinearStreamingTttNope => 3,
    }
}

fn mini_transformer_binary_position_code(position: MiniTransformerPositionPolicy) -> u8 {
    match position {
        MiniTransformerPositionPolicy::LearnedAbsolute => 0,
        MiniTransformerPositionPolicy::Nope => 1,
    }
}

fn mini_transformer_binary_config_flags(config: MiniTransformerMlpTrainConfig) -> u16 {
    let mut flags = 0_u16;
    if config.adaptive_rule_shifts {
        flags |= 1 << 0;
    }
    if config.adaptive_attention_shifts {
        flags |= 1 << 1;
    }
    if config.adaptive_holographic_shifts {
        flags |= 1 << 2;
    }
    if config.attention_vo_error_feedback {
        flags |= 1 << 3;
    }
    if config.attention_vo_oracle {
        flags |= 1 << 4;
    }
    if config.reject_loss_regression {
        flags |= 1 << 5;
    }
    if config.batch_mode == MiniTransformerBatchMode::MapReduce {
        flags |= 1 << 6;
    }
    flags
}

fn mini_transformer_binary_component_code(component: &str) -> u8 {
    match component {
        "output" | "output_head" => 0,
        "mlp" => 1,
        "embedding" => 2,
        "attention" => 3,
        "attention_q" => 4,
        "attention_qk" | "attention_k" => 5,
        _ => u8::MAX,
    }
}

fn mini_transformer_binary_reason_code(reason: &str) -> u8 {
    match reason {
        "rollback" | "rejected" => 0,
        "saturation" => 1,
        "zero_delta" | "dead_component" => 2,
        "movement" | "active_delta" => 3,
        "holographic" | "holographic_advisory" => 4,
        _ => u8::MAX,
    }
}

fn push_binary_optional_u32(out: &mut Vec<u8>, value: Option<usize>) {
    match value {
        Some(value) => push_binary_u32_clamped(out, value),
        None => push_binary_u32(out, u32::MAX),
    }
}

fn push_binary_u16_clamped(out: &mut Vec<u8>, value: usize) {
    push_binary_u16(out, value.min(usize::from(u16::MAX)) as u16);
}

fn push_binary_u32_clamped(out: &mut Vec<u8>, value: usize) {
    push_binary_u32(out, value.min(u32::MAX as usize) as u32);
}

fn push_binary_u32_saturating(out: &mut Vec<u8>, value: u64) {
    push_binary_u32(out, value.min(u64::from(u32::MAX)) as u32);
}

fn push_binary_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

impl MiniTransformerMlpTrainingProgressTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            "nsrl.training_mini_transformer_progress.v1",
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", MINI_TRANSFORMER_MLP_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        comma(&mut out);
        push_hash_field(&mut out, "window_hash", self.window_hash);
        comma(&mut out);
        push_usize_field(&mut out, "windows", self.windows);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(&mut out, "accepted_batch_count", self.accepted_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rejected_batch_count", self.rejected_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rollback_count", self.rollback_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rejected_window_count",
            self.rejected_window_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "output_head_delta_l1", self.output_head_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_delta_l1", self.mlp_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_delta_l1", self.embedding_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_delta_l1", self.attention_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_delta_l1", self.attention_q_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_delta_l1", self.attention_k_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_delta_l1", self.attention_v_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_delta_l1", self.attention_o_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "output_head_carry_l1", self.output_head_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_carry_l1", self.mlp_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_carry_l1", self.embedding_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_carry_l1", self.attention_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_carry_l1", self.attention_q_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_carry_l1", self.attention_k_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_carry_l1", self.attention_v_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_carry_l1", self.attention_o_carry_l1);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_shift_adjustment_count",
            self.adaptive_rule_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_shift_adjustment_count",
            self.adaptive_holographic_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_output_learning_rate_shift",
            usize::from(self.current_output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_mlp_learning_rate_shift",
            usize::from(self.current_mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_embedding_learning_rate_shift",
            usize::from(self.current_embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_learning_rate_shift",
            usize::from(self.current_attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_q_learning_rate_shift",
            usize::from(self.current_attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_qk_learning_rate_shift",
            usize::from(self.current_attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpModel {
    pub fn new_initial() -> Self {
        Self::new_initial_with_seq_len(DEFAULT_MINI_TRANSFORMER_SEQ_LEN)
    }

    pub fn new_initial_with_seq_len(context_seq_len: usize) -> Self {
        Self::new_initial_with_seq_len_and_layers(context_seq_len, DEFAULT_MINI_TRANSFORMER_LAYERS)
            .expect("default mini transformer layer count should be valid")
    }

    pub fn new_initial_with_seq_len_and_layers(
        context_seq_len: usize,
        layers: usize,
    ) -> Result<Self, TrainError> {
        if layers == 0 {
            return Err(TrainError::InvalidModel("bad mini transformer layer count"));
        }
        Self {
            context_seq_len,
            embeddings: initial_mini_transformer_embeddings(),
            position_embeddings: initial_mini_transformer_position_embeddings(context_seq_len),
            attention_rms_weights: Vec::new(),
            mlp_rms_weights: Vec::new(),
            q_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            k_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            v_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            o_weights: stack_i8_layers_with_active_final(
                vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            up_weights: stack_i8_layers_with_active_final(
                initial_mini_transformer_mlp_up_weights(),
                initial_mini_transformer_mlp_up_weights(),
                layers,
            ),
            gate_weights: stack_i8_layers_with_active_final(
                initial_mini_transformer_mlp_gate_weights(),
                initial_mini_transformer_mlp_gate_weights(),
                layers,
            ),
            down_weights: stack_i8_layers_with_active_final(
                vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
                initial_mini_transformer_mlp_down_weights(),
                layers,
            ),
            output_weights: initial_mini_transformer_output_weights(),
        }
        .validate()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_seq_len: usize,
        embeddings: Vec<i16>,
        position_embeddings: Vec<i16>,
        q_weights: Vec<i8>,
        k_weights: Vec<i8>,
        v_weights: Vec<i8>,
        o_weights: Vec<i8>,
        up_weights: Vec<i8>,
        gate_weights: Vec<i8>,
        down_weights: Vec<i8>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        Self::new_with_rms_weights(
            context_seq_len,
            embeddings,
            position_embeddings,
            Vec::new(),
            Vec::new(),
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_rms_weights(
        context_seq_len: usize,
        embeddings: Vec<i16>,
        position_embeddings: Vec<i16>,
        attention_rms_weights: Vec<i16>,
        mlp_rms_weights: Vec<i16>,
        q_weights: Vec<i8>,
        k_weights: Vec<i8>,
        v_weights: Vec<i8>,
        o_weights: Vec<i8>,
        up_weights: Vec<i8>,
        gate_weights: Vec<i8>,
        down_weights: Vec<i8>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        if context_seq_len == 0 {
            return Err(TrainError::InvalidModel(
                "bad mini transformer context seq_len",
            ));
        }
        if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer embedding count",
            ));
        }
        if position_embeddings.len()
            != context_seq_len
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidModel(
                    "mini transformer position count overflow",
                ))?
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer position embedding count",
            ));
        }
        let attention_weight_count = mini_transformer_attention_weight_count()?;
        let mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        let attention_layers = infer_layer_count(q_weights.len(), attention_weight_count).ok_or(
            TrainError::InvalidModel("wrong mini transformer attention weight count"),
        )?;
        let mlp_layers = infer_layer_count(up_weights.len(), mlp_up_or_gate_count).ok_or(
            TrainError::InvalidModel("wrong mini transformer up/gate weight count"),
        )?;
        if attention_layers == 0 || mlp_layers == 0 || attention_layers != mlp_layers {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer layer count",
            ));
        }
        let expected_rms_weight_count = attention_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm weight count overflow"))?;
        let rms_disabled = attention_rms_weights.is_empty() && mlp_rms_weights.is_empty();
        let rms_enabled = attention_rms_weights.len() == expected_rms_weight_count
            && mlp_rms_weights.len() == expected_rms_weight_count;
        if !rms_disabled && !rms_enabled {
            return Err(TrainError::InvalidModel("wrong RMSNorm weight count"));
        }
        if k_weights.len() != q_weights.len()
            || v_weights.len() != q_weights.len()
            || o_weights.len() != q_weights.len()
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer attention weight count",
            ));
        }
        if gate_weights.len() != up_weights.len() {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer up/gate weight count",
            ));
        }
        if down_weights.len() != mlp_layers * mlp_down_count {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer down weight count",
            ));
        }
        if output_weights.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer output weight count",
            ));
        }

        Ok(Self {
            context_seq_len,
            embeddings,
            position_embeddings,
            attention_rms_weights,
            mlp_rms_weights,
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        })
    }

    fn validate(self) -> Result<Self, TrainError> {
        Self::new_with_rms_weights(
            self.context_seq_len,
            self.embeddings,
            self.position_embeddings,
            self.attention_rms_weights,
            self.mlp_rms_weights,
            self.q_weights,
            self.k_weights,
            self.v_weights,
            self.o_weights,
            self.up_weights,
            self.gate_weights,
            self.down_weights,
            self.output_weights,
        )
    }

    pub fn transformer_layers(&self) -> usize {
        let Ok(attention_weight_count) = mini_transformer_attention_weight_count() else {
            return 0;
        };
        let Some(attention_layers) =
            infer_layer_count(self.q_weights.len(), attention_weight_count)
        else {
            return 0;
        };
        attention_layers
    }

    fn checked_transformer_layers(&self) -> Result<usize, TrainError> {
        let layers = self.transformer_layers();
        if layers == 0 {
            return Err(TrainError::InvalidModel("bad mini transformer layer count"));
        }
        let attention_weight_count = mini_transformer_attention_weight_count()?;
        let mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        if self.k_weights.len() != self.q_weights.len()
            || self.v_weights.len() != self.q_weights.len()
            || self.o_weights.len() != self.q_weights.len()
            || self.up_weights.len() != layers * mlp_up_or_gate_count
            || self.gate_weights.len() != self.up_weights.len()
            || self.down_weights.len() != layers * mlp_down_count
            || self.q_weights.len() != layers * attention_weight_count
            || (!self.attention_rms_weights.is_empty()
                && (self.attention_rms_weights.len() != layers * MINI_TRANSFORMER_D_MODEL
                    || self.mlp_rms_weights.len() != layers * MINI_TRANSFORMER_D_MODEL))
            || (self.attention_rms_weights.is_empty() != self.mlp_rms_weights.is_empty())
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer layer tensor count",
            ));
        }
        Ok(layers)
    }

    pub fn rms_norm_enabled(&self) -> bool {
        !self.attention_rms_weights.is_empty()
    }

    pub fn enable_rms_norm(&mut self) -> Result<(), TrainError> {
        if self.rms_norm_enabled() {
            return Ok(());
        }
        self.enable_rms_norm_with_gamma(DEFAULT_MINI_TRANSFORMER_RMS_GAMMA_Q15)
    }

    pub fn enable_rms_norm_with_gamma(&mut self, gamma_q15: i16) -> Result<(), TrainError> {
        if gamma_q15 <= 0 {
            return Err(TrainError::InvalidConfig);
        }
        let layers = self.checked_transformer_layers()?;
        if self.rms_norm_enabled() {
            if self
                .attention_rms_weights
                .iter()
                .chain(self.mlp_rms_weights.iter())
                .all(|&weight| weight == gamma_q15)
            {
                return Ok(());
            }
            return Err(TrainError::InvalidConfig);
        }
        let count = layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm weight count overflow"))?;
        self.attention_rms_weights = vec![gamma_q15; count];
        self.mlp_rms_weights = vec![gamma_q15; count];
        Ok(())
    }

    fn rms_weight_range(&self, layer_index: usize) -> Result<Range<usize>, TrainError> {
        if !self.rms_norm_enabled() || layer_index >= self.checked_transformer_layers()? {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, MINI_TRANSFORMER_D_MODEL)
    }

    fn attention_weight_range(&self, layer_index: usize) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_attention_weight_count()?)
    }

    fn mlp_up_or_gate_weight_range(&self, layer_index: usize) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_mlp_up_or_gate_weight_count()?)
    }

    fn mlp_down_weight_range(&self, layer_index: usize) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_mlp_down_weight_count()?)
    }

    fn final_attention_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.attention_weight_range(layers - 1)
    }

    fn final_mlp_up_or_gate_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.mlp_up_or_gate_weight_range(layers - 1)
    }

    fn final_mlp_down_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.mlp_down_weight_range(layers - 1)
    }

    pub fn embedding_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i16_slice(&self.position_embeddings);
        hasher.finish()
    }

    pub fn attention_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_i8_slice(&self.q_weights);
        hasher.update_i8_slice(&self.k_weights);
        hasher.update_i8_slice(&self.v_weights);
        hasher.update_i8_slice(&self.o_weights);
        hasher.finish()
    }

    pub fn attention_q_hash(&self) -> u64 {
        hash_i8_slice(&self.q_weights)
    }

    pub fn attention_k_hash(&self) -> u64 {
        hash_i8_slice(&self.k_weights)
    }

    pub fn attention_v_hash(&self) -> u64 {
        hash_i8_slice(&self.v_weights)
    }

    pub fn attention_o_hash(&self) -> u64 {
        hash_i8_slice(&self.o_weights)
    }

    pub fn mlp_hash(&self) -> u64 {
        hash_three_i8_slices(&self.up_weights, &self.gate_weights, &self.down_weights)
    }

    pub fn output_head_hash(&self) -> u64 {
        hash_i8_slice(&self.output_weights)
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.context_seq_len);
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i16_slice(&self.position_embeddings);
        if self.rms_norm_enabled() {
            hasher.update_i16_slice(&self.attention_rms_weights);
            hasher.update_i16_slice(&self.mlp_rms_weights);
        }
        hasher.update_i8_slice(&self.q_weights);
        hasher.update_i8_slice(&self.k_weights);
        hasher.update_i8_slice(&self.v_weights);
        hasher.update_i8_slice(&self.o_weights);
        hasher.update_i8_slice(&self.up_weights);
        hasher.update_i8_slice(&self.gate_weights);
        hasher.update_i8_slice(&self.down_weights);
        hasher.update_i8_slice(&self.output_weights);
        hasher.finish()
    }

    pub fn optimizer_parameter_count(&self) -> Result<usize, TrainError> {
        [
            self.embeddings.len(),
            self.position_embeddings.len(),
            self.attention_rms_weights.len(),
            self.mlp_rms_weights.len(),
            self.q_weights.len(),
            self.k_weights.len(),
            self.v_weights.len(),
            self.o_weights.len(),
            self.up_weights.len(),
            self.gate_weights.len(),
            self.down_weights.len(),
            self.output_weights.len(),
        ]
        .into_iter()
        .try_fold(0_usize, |total, count| {
            total.checked_add(count).ok_or(TrainError::InvalidModel(
                "optimizer parameter count overflow",
            ))
        })
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        let embedding_bytes = checked_i16_tensor_bytes(
            self.embeddings.len(),
            "mini transformer embedding bytes overflow",
        )?;
        let position_embedding_bytes = checked_i16_tensor_bytes(
            self.position_embeddings.len(),
            "mini transformer position embedding bytes overflow",
        )?;
        let attention_rms_bytes = checked_i16_tensor_bytes(
            self.attention_rms_weights.len(),
            "mini transformer attention RMS bytes overflow",
        )?;
        let mlp_rms_bytes = checked_i16_tensor_bytes(
            self.mlp_rms_weights.len(),
            "mini transformer MLP RMS bytes overflow",
        )?;
        let weight_bytes = checked_model_capacity(
            0,
            &[
                self.q_weights.len(),
                self.k_weights.len(),
                self.v_weights.len(),
                self.o_weights.len(),
                self.up_weights.len(),
                self.gate_weights.len(),
                self.down_weights.len(),
                self.output_weights.len(),
            ],
        )?;
        let mut out = Vec::with_capacity(checked_model_capacity(
            136,
            &[
                embedding_bytes,
                position_embedding_bytes,
                weight_bytes,
                attention_rms_bytes,
                mlp_rms_bytes,
            ],
        )?);
        out.extend_from_slice(MINI_TRANSFORMER_MODEL_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(
                MINI_TRANSFORMER_D_MODEL,
                "mini transformer d_model exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "mini transformer heads exceeds u32")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                MINI_TRANSFORMER_HIDDEN_DIM,
                "mini transformer hidden_dim exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.context_seq_len,
                "mini transformer context_seq_len exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.embeddings.len(),
                "mini transformer embedding count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.position_embeddings.len(),
                "mini transformer position embedding count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.q_weights.len(), "mini transformer q count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.k_weights.len(), "mini transformer k count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.v_weights.len(), "mini transformer v count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.o_weights.len(), "mini transformer o count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.up_weights.len(),
                "mini transformer up count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.gate_weights.len(),
                "mini transformer gate count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.down_weights.len(),
                "mini transformer down count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.output_weights.len(),
                "mini transformer output count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(&self.embedding_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_q_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_k_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_v_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_o_hash().to_le_bytes());
        out.extend_from_slice(&self.mlp_hash().to_le_bytes());
        out.extend_from_slice(&self.output_head_hash().to_le_bytes());
        out.extend_from_slice(&self.model_hash().to_le_bytes());
        for &embedding in self.embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        for &embedding in self.position_embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        out.extend(self.q_weights.iter().map(|&weight| weight as u8));
        out.extend(self.k_weights.iter().map(|&weight| weight as u8));
        out.extend(self.v_weights.iter().map(|&weight| weight as u8));
        out.extend(self.o_weights.iter().map(|&weight| weight as u8));
        out.extend(self.up_weights.iter().map(|&weight| weight as u8));
        out.extend(self.gate_weights.iter().map(|&weight| weight as u8));
        out.extend(self.down_weights.iter().map(|&weight| weight as u8));
        out.extend(self.output_weights.iter().map(|&weight| weight as u8));
        for &weight in &self.attention_rms_weights {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        for &weight in &self.mlp_rms_weights {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer model should fit on-disk format")
    }

    /// Returns the model hash recorded in a serialized V4 or V5 artifact.
    ///
    /// This is intentionally separate from [`Self::model_hash`]: loading a
    /// historical 32-wide V4 artifact upgrades it to the current geometry, so
    /// the in-memory model has a new hash after the source artifact has been
    /// authenticated.
    pub fn serialized_model_hash(bytes: &[u8]) -> Result<u64, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        let magic = &bytes[..MINI_TRANSFORMER_MODEL_MAGIC.len()];
        if magic != MINI_TRANSFORMER_MODEL_MAGIC && magic != MINI_TRANSFORMER_LEGACY_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }
        let hash_offset = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 7;
        let hash_bytes = bytes
            .get(hash_offset..hash_offset + 8)
            .ok_or(TrainError::InvalidModel("artifact too short"))?;
        Ok(u64::from_le_bytes(
            hash_bytes
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad model hash"))?,
        ))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        let magic = &bytes[..MINI_TRANSFORMER_MODEL_MAGIC.len()];
        let legacy = magic == MINI_TRANSFORMER_LEGACY_MODEL_MAGIC;
        if magic != MINI_TRANSFORMER_MODEL_MAGIC && !legacy {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = MINI_TRANSFORMER_MODEL_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let context_seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let position_embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let q_count = read_u64_le(bytes, &mut offset)? as usize;
        let k_count = read_u64_le(bytes, &mut offset)? as usize;
        let v_count = read_u64_le(bytes, &mut offset)? as usize;
        let o_count = read_u64_le(bytes, &mut offset)? as usize;
        let up_count = read_u64_le(bytes, &mut offset)? as usize;
        let gate_count = read_u64_le(bytes, &mut offset)? as usize;
        let down_count = read_u64_le(bytes, &mut offset)? as usize;
        let output_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_embedding_hash = read_u64_le(bytes, &mut offset)?;
        let expected_q_hash = read_u64_le(bytes, &mut offset)?;
        let expected_k_hash = read_u64_le(bytes, &mut offset)?;
        let expected_v_hash = read_u64_le(bytes, &mut offset)?;
        let expected_o_hash = read_u64_le(bytes, &mut offset)?;
        let expected_mlp_hash = read_u64_le(bytes, &mut offset)?;
        let expected_output_hash = read_u64_le(bytes, &mut offset)?;
        let expected_model_hash = read_u64_le(bytes, &mut offset)?;

        if legacy
            && d_model == MINI_TRANSFORMER_LEGACY_V4_D_MODEL
            && heads == MINI_TRANSFORMER_LEGACY_V4_HEADS
            && hidden_dim == MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM
        {
            return decode_and_upgrade_legacy_v4_model(
                bytes,
                offset,
                vocab,
                context_seq_len,
                embedding_count,
                position_embedding_count,
                q_count,
                k_count,
                v_count,
                o_count,
                up_count,
                gate_count,
                down_count,
                output_count,
                expected_embedding_hash,
                expected_q_hash,
                expected_k_hash,
                expected_v_hash,
                expected_o_hash,
                expected_mlp_hash,
                expected_output_hash,
                expected_model_hash,
            );
        }

        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }

        let expected_attention_count = mini_transformer_attention_weight_count()?;
        let expected_mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let expected_mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        let inferred_attention_layers = infer_layer_count(q_count, expected_attention_count)
            .ok_or(TrainError::InvalidModel("attention tensor count mismatch"))?;
        let inferred_mlp_layers = infer_layer_count(up_count, expected_mlp_up_or_gate_count)
            .ok_or(TrainError::InvalidModel("mlp tensor count mismatch"))?;
        let expected_position_embedding_count = context_seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel(
                "position embedding count overflow",
            ))?;
        if embedding_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
            || position_embedding_count != expected_position_embedding_count
            || inferred_attention_layers == 0
            || inferred_mlp_layers == 0
            || inferred_attention_layers != inferred_mlp_layers
            || k_count != q_count
            || v_count != q_count
            || o_count != q_count
            || gate_count != up_count
            || down_count != inferred_mlp_layers * expected_mlp_down_count
            || output_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        {
            return Err(TrainError::InvalidModel("tensor count mismatch"));
        }

        let embedding_bytes = embedding_count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("embedding length overflow"))?;
        let position_embedding_bytes =
            position_embedding_count
                .checked_mul(2)
                .ok_or(TrainError::InvalidModel(
                    "position embedding length overflow",
                ))?;
        let weight_bytes = q_count
            .checked_add(k_count)
            .and_then(|value| value.checked_add(v_count))
            .and_then(|value| value.checked_add(o_count))
            .and_then(|value| value.checked_add(up_count))
            .and_then(|value| value.checked_add(gate_count))
            .and_then(|value| value.checked_add(down_count))
            .and_then(|value| value.checked_add(output_count))
            .ok_or(TrainError::InvalidModel("weight length overflow"))?;
        let base_expected_len = offset
            .checked_add(embedding_bytes)
            .and_then(|value| value.checked_add(position_embedding_bytes))
            .and_then(|value| value.checked_add(weight_bytes))
            .ok_or(TrainError::InvalidModel("artifact length overflow"))?;
        let rms_count = inferred_attention_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm count overflow"))?;
        let rms_bytes = rms_count
            .checked_mul(4)
            .ok_or(TrainError::InvalidModel("RMSNorm bytes overflow"))?;
        let rms_expected_len = base_expected_len
            .checked_add(rms_bytes)
            .ok_or(TrainError::InvalidModel("artifact length overflow"))?;
        let rms_enabled = !legacy && bytes.len() == rms_expected_len;
        if (legacy && bytes.len() != base_expected_len)
            || (!legacy && bytes.len() != base_expected_len && !rms_enabled)
        {
            return Err(TrainError::InvalidModel("artifact length mismatch"));
        }

        let embedding_end = offset + embedding_bytes;
        let mut embeddings = Vec::with_capacity(embedding_count);
        for chunk in bytes[offset..embedding_end].chunks_exact(2) {
            embeddings.push(i16::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TrainError::InvalidModel("bad embedding"))?,
            ));
        }
        offset = embedding_end;

        let position_embedding_end = offset + position_embedding_bytes;
        let mut position_embeddings = Vec::with_capacity(position_embedding_count);
        for chunk in bytes[offset..position_embedding_end].chunks_exact(2) {
            position_embeddings.push(i16::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TrainError::InvalidModel("bad position embedding"))?,
            ));
        }
        offset = position_embedding_end;

        let q_weights = read_i8_vec(bytes, &mut offset, q_count)?;
        let k_weights = read_i8_vec(bytes, &mut offset, k_count)?;
        let v_weights = read_i8_vec(bytes, &mut offset, v_count)?;
        let o_weights = read_i8_vec(bytes, &mut offset, o_count)?;
        let up_weights = read_i8_vec(bytes, &mut offset, up_count)?;
        let gate_weights = read_i8_vec(bytes, &mut offset, gate_count)?;
        let down_weights = read_i8_vec(bytes, &mut offset, down_count)?;
        let output_weights = read_i8_vec(bytes, &mut offset, output_count)?;
        let (attention_rms_weights, mlp_rms_weights) = if rms_enabled {
            (
                read_i16_vec(bytes, &mut offset, rms_count)?,
                read_i16_vec(bytes, &mut offset, rms_count)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let model = Self::new_with_rms_weights(
            context_seq_len,
            embeddings,
            position_embeddings,
            attention_rms_weights,
            mlp_rms_weights,
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        )?;
        if model.embedding_hash() != expected_embedding_hash {
            return Err(TrainError::InvalidModel("embedding hash mismatch"));
        }
        if model.attention_q_hash() != expected_q_hash
            || model.attention_k_hash() != expected_k_hash
            || model.attention_v_hash() != expected_v_hash
            || model.attention_o_hash() != expected_o_hash
        {
            return Err(TrainError::InvalidModel("attention hash mismatch"));
        }
        if model.mlp_hash() != expected_mlp_hash {
            return Err(TrainError::InvalidModel("mlp hash mismatch"));
        }
        if model.output_head_hash() != expected_output_hash {
            return Err(TrainError::InvalidModel("output hash mismatch"));
        }
        if model.model_hash() != expected_model_hash {
            return Err(TrainError::InvalidModel("model hash mismatch"));
        }
        Ok(model)
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_and_upgrade_legacy_v4_model(
    bytes: &[u8],
    mut offset: usize,
    vocab: usize,
    context_seq_len: usize,
    embedding_count: usize,
    position_embedding_count: usize,
    q_count: usize,
    k_count: usize,
    v_count: usize,
    o_count: usize,
    up_count: usize,
    gate_count: usize,
    down_count: usize,
    output_count: usize,
    expected_embedding_hash: u64,
    expected_q_hash: u64,
    expected_k_hash: u64,
    expected_v_hash: u64,
    expected_o_hash: u64,
    expected_mlp_hash: u64,
    expected_output_hash: u64,
    expected_model_hash: u64,
) -> Result<MiniTransformerMlpModel, TrainError> {
    let expected_embedding_count = BYTE_VOCAB
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy embedding count overflow"))?;
    let expected_position_count = context_seq_len
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy position count overflow"))?;
    let expected_attention_count = MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy attention count overflow"))?;
    let expected_up_count = MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM)
        .ok_or(TrainError::InvalidModel("legacy MLP count overflow"))?;
    let expected_down_count = MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy MLP count overflow"))?;
    let expected_output_count = BYTE_VOCAB
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy output count overflow"))?;
    if vocab != BYTE_VOCAB
        || context_seq_len == 0
        || embedding_count != expected_embedding_count
        || position_embedding_count != expected_position_count
        || q_count != expected_attention_count
        || k_count != expected_attention_count
        || v_count != expected_attention_count
        || o_count != expected_attention_count
        || up_count != expected_up_count
        || gate_count != expected_up_count
        || down_count != expected_down_count
        || output_count != expected_output_count
    {
        return Err(TrainError::InvalidModel("legacy V4 tensor count mismatch"));
    }

    let expected_len = offset
        .checked_add(
            embedding_count
                .checked_mul(2)
                .ok_or(TrainError::InvalidModel("legacy embedding length overflow"))?,
        )
        .and_then(|value| value.checked_add(position_embedding_count.checked_mul(2)?))
        .and_then(|value| value.checked_add(q_count))
        .and_then(|value| value.checked_add(k_count))
        .and_then(|value| value.checked_add(v_count))
        .and_then(|value| value.checked_add(o_count))
        .and_then(|value| value.checked_add(up_count))
        .and_then(|value| value.checked_add(gate_count))
        .and_then(|value| value.checked_add(down_count))
        .and_then(|value| value.checked_add(output_count))
        .ok_or(TrainError::InvalidModel("legacy artifact length overflow"))?;
    if bytes.len() != expected_len {
        return Err(TrainError::InvalidModel(
            "legacy V4 artifact length mismatch",
        ));
    }

    let embeddings = read_i16_vec(bytes, &mut offset, embedding_count)?;
    let position_embeddings = read_i16_vec(bytes, &mut offset, position_embedding_count)?;
    let q_weights = read_i8_vec(bytes, &mut offset, q_count)?;
    let k_weights = read_i8_vec(bytes, &mut offset, k_count)?;
    let v_weights = read_i8_vec(bytes, &mut offset, v_count)?;
    let o_weights = read_i8_vec(bytes, &mut offset, o_count)?;
    let up_weights = read_i8_vec(bytes, &mut offset, up_count)?;
    let gate_weights = read_i8_vec(bytes, &mut offset, gate_count)?;
    let down_weights = read_i8_vec(bytes, &mut offset, down_count)?;
    let output_weights = read_i8_vec(bytes, &mut offset, output_count)?;

    let mut embedding_hasher = StableHasher::new();
    embedding_hasher.update_i16_slice(&embeddings);
    embedding_hasher.update_i16_slice(&position_embeddings);
    if embedding_hasher.finish() != expected_embedding_hash {
        return Err(TrainError::InvalidModel("embedding hash mismatch"));
    }
    if hash_i8_slice(&q_weights) != expected_q_hash
        || hash_i8_slice(&k_weights) != expected_k_hash
        || hash_i8_slice(&v_weights) != expected_v_hash
        || hash_i8_slice(&o_weights) != expected_o_hash
    {
        return Err(TrainError::InvalidModel("attention hash mismatch"));
    }
    if hash_three_i8_slices(&up_weights, &gate_weights, &down_weights) != expected_mlp_hash {
        return Err(TrainError::InvalidModel("mlp hash mismatch"));
    }
    if hash_i8_slice(&output_weights) != expected_output_hash {
        return Err(TrainError::InvalidModel("output hash mismatch"));
    }
    let mut model_hasher = StableHasher::new();
    model_hasher.update_usize(context_seq_len);
    model_hasher.update_i16_slice(&embeddings);
    model_hasher.update_i16_slice(&position_embeddings);
    model_hasher.update_i8_slice(&q_weights);
    model_hasher.update_i8_slice(&k_weights);
    model_hasher.update_i8_slice(&v_weights);
    model_hasher.update_i8_slice(&o_weights);
    model_hasher.update_i8_slice(&up_weights);
    model_hasher.update_i8_slice(&gate_weights);
    model_hasher.update_i8_slice(&down_weights);
    model_hasher.update_i8_slice(&output_weights);
    if model_hasher.finish() != expected_model_hash {
        return Err(TrainError::InvalidModel("model hash mismatch"));
    }

    upgrade_legacy_v4_model(
        context_seq_len,
        embeddings,
        position_embeddings,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        up_weights,
        gate_weights,
        down_weights,
        output_weights,
    )
}

#[allow(clippy::too_many_arguments)]
fn upgrade_legacy_v4_model(
    context_seq_len: usize,
    embeddings: Vec<i16>,
    position_embeddings: Vec<i16>,
    q_weights: Vec<i8>,
    k_weights: Vec<i8>,
    v_weights: Vec<i8>,
    o_weights: Vec<i8>,
    up_weights: Vec<i8>,
    gate_weights: Vec<i8>,
    down_weights: Vec<i8>,
    output_weights: Vec<i8>,
) -> Result<MiniTransformerMlpModel, TrainError> {
    if MINI_TRANSFORMER_D_MODEL != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * 4
        || (MINI_TRANSFORMER_HEADS != MINI_TRANSFORMER_LEGACY_V4_HEADS
            && MINI_TRANSFORMER_HEADS != MINI_TRANSFORMER_LEGACY_V4_HEADS * 4)
        || MINI_TRANSFORMER_HIDDEN_DIM != MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * 4
    {
        return Err(TrainError::InvalidModel(
            "unsupported legacy V4 geometry upgrade",
        ));
    }
    let embeddings = widen_legacy_model_rows_i16(&embeddings, BYTE_VOCAB)?;
    let position_embeddings = widen_legacy_model_rows_i16(&position_embeddings, context_seq_len)?;
    let q_weights = widen_legacy_model_matrix(&q_weights, 2)?;
    let k_weights = widen_legacy_model_matrix(&k_weights, 2)?;
    let v_weights = widen_legacy_model_matrix(&v_weights, 4)?;
    let o_weights = widen_legacy_model_matrix(&o_weights, 4)?;
    let up_weights = widen_legacy_up_or_gate_matrix(&up_weights)?;
    let gate_weights = widen_legacy_up_or_gate_matrix(&gate_weights)?;
    let down_weights = widen_legacy_down_matrix(&down_weights)?;
    let output_weights = widen_legacy_output_matrix(&output_weights)?;
    MiniTransformerMlpModel::new(
        context_seq_len,
        embeddings,
        position_embeddings,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        up_weights,
        gate_weights,
        down_weights,
        output_weights,
    )
}

fn legacy_model_dim_index(index: usize, replica: usize) -> Result<usize, TrainError> {
    let old_head_dim = MINI_TRANSFORMER_LEGACY_V4_D_MODEL / MINI_TRANSFORMER_LEGACY_V4_HEADS;
    let new_head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    if index >= MINI_TRANSFORMER_LEGACY_V4_D_MODEL || replica >= 4 {
        return Err(TrainError::InvalidModel("legacy model index out of range"));
    }
    let head = index / old_head_dim;
    let dim = index % old_head_dim;
    if MINI_TRANSFORMER_HEADS == MINI_TRANSFORMER_LEGACY_V4_HEADS {
        Ok(head * new_head_dim + dim * 4 + replica)
    } else if MINI_TRANSFORMER_HEADS == MINI_TRANSFORMER_LEGACY_V4_HEADS * 4
        && new_head_dim == old_head_dim
    {
        Ok((head * 4 + replica) * new_head_dim + dim)
    } else {
        Err(TrainError::InvalidModel(
            "unsupported legacy model head mapping",
        ))
    }
}

fn widen_legacy_model_rows_i16(values: &[i16], rows: usize) -> Result<Vec<i16>, TrainError> {
    if values.len() != rows * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy row tensor mismatch"));
    }
    let mut out = vec![0_i16; rows * MINI_TRANSFORMER_D_MODEL];
    for row in 0..rows {
        for old_dim in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
            let value = values[row * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_dim];
            for replica in 0..4 {
                out[row * MINI_TRANSFORMER_D_MODEL + legacy_model_dim_index(old_dim, replica)?] =
                    value;
            }
        }
    }
    Ok(out)
}

fn widen_legacy_model_matrix(values: &[i8], output_replicas: usize) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        || !(1..=4).contains(&output_replicas)
    {
        return Err(TrainError::InvalidModel("legacy attention tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        for output_replica in 0..output_replicas {
            let new_output = legacy_model_dim_index(old_output, output_replica)?;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
                let new_input = legacy_model_dim_index(old_input, 0)?;
                out[new_output * MINI_TRANSFORMER_D_MODEL + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_up_or_gate_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy MLP tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
        for output_replica in 0..4 {
            let new_output = old_output * 4 + output_replica;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
                let new_input = legacy_model_dim_index(old_input, 0)?;
                out[new_output * MINI_TRANSFORMER_D_MODEL + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_down_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
        return Err(TrainError::InvalidModel("legacy MLP tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        for output_replica in 0..4 {
            let new_output = legacy_model_dim_index(old_output, output_replica)?;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
                let new_input = old_input * 4;
                out[new_output * MINI_TRANSFORMER_HIDDEN_DIM + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_output_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy output tensor mismatch"));
    }
    let mut out = vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    for output in 0..BYTE_VOCAB {
        for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
            let new_input = legacy_model_dim_index(old_input, 0)?;
            out[output * MINI_TRANSFORMER_D_MODEL + new_input] =
                values[output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
        }
    }
    Ok(out)
}

impl MiniTransformerBlockLowRankExpert {
    pub fn new_for_model(
        model: &MiniTransformerMlpModel,
        rank: usize,
        projection_seed: u64,
    ) -> Result<Self, TrainError> {
        Self::new_for_model_with_residual_shift(model, rank, projection_seed, 0)
    }

    pub fn new_for_model_with_residual_shift(
        model: &MiniTransformerMlpModel,
        rank: usize,
        projection_seed: u64,
        residual_shift: u8,
    ) -> Result<Self, TrainError> {
        let transformer_layers = model.checked_transformer_layers()?;
        if rank == 0 || rank > MINI_TRANSFORMER_D_MODEL || residual_shift > 15 {
            return Err(TrainError::InvalidConfig);
        }
        let parameter_count = transformer_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .and_then(|value| value.checked_mul(rank))
            .ok_or(TrainError::InvalidConfig)?;
        Ok(Self {
            trunk_model_hash: model.model_hash(),
            transformer_layers,
            rank,
            projection_seed,
            residual_shift,
            expansion_weights_q15: vec![0_i16; parameter_count],
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.expansion_weights_q15.len()
    }

    pub fn validate_for_model(&self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let expected = self
            .transformer_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .and_then(|value| value.checked_mul(self.rank))
            .ok_or(TrainError::InvalidConfig)?;
        if self.trunk_model_hash != model.model_hash()
            || self.transformer_layers != model.checked_transformer_layers()?
            || self.rank == 0
            || self.rank > MINI_TRANSFORMER_D_MODEL
            || self.residual_shift > 15
            || self.expansion_weights_q15.len() != expected
        {
            return Err(TrainError::InvalidModel("block expert/model mismatch"));
        }
        Ok(())
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.transformer_layers == 0
            || self.rank == 0
            || self.rank > MINI_TRANSFORMER_D_MODEL
            || self.residual_shift > 15
            || self.expansion_weights_q15.len()
                != self
                    .transformer_layers
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_mul(self.rank))
                    .ok_or(TrainError::InvalidConfig)?
        {
            return Err(TrainError::InvalidModel("invalid block expert"));
        }
        let mut out = Vec::with_capacity(80 + self.expansion_weights_q15.len() * 2);
        out.extend_from_slice(MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_D_MODEL, "d_model exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "heads exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HIDDEN_DIM, "hidden_dim exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        push_model_usize(&mut out, self.transformer_layers, "layers exceed u64")?;
        push_model_usize(&mut out, self.rank, "rank exceeds u64")?;
        out.extend_from_slice(&self.projection_seed.to_le_bytes());
        out.push(self.residual_shift);
        out.extend_from_slice(&[0_u8; 7]);
        push_model_usize(
            &mut out,
            self.expansion_weights_q15.len(),
            "block expert parameters exceed u64",
        )?;
        for &weight in &self.expansion_weights_q15 {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        let checksum = hash_u8_slice(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("valid block expert should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        const HEADER_WITH_CHECKSUM: usize = 80;
        if bytes.len() < HEADER_WITH_CHECKSUM
            || &bytes[..MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC.len()]
                != MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC
        {
            return Err(TrainError::InvalidModel("bad block expert artifact"));
        }
        let checksum_offset = bytes.len() - 8;
        let mut checksum_cursor = checksum_offset;
        let checksum = read_u64_le(bytes, &mut checksum_cursor)?;
        if checksum_cursor != bytes.len() || hash_u8_slice(&bytes[..checksum_offset]) != checksum {
            return Err(TrainError::InvalidModel("block expert checksum mismatch"));
        }
        let mut offset = MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let trunk_model_hash = read_u64_le(bytes, &mut offset)?;
        let transformer_layers = read_model_usize(bytes, &mut offset)?;
        let rank = read_model_usize(bytes, &mut offset)?;
        let projection_seed = read_u64_le(bytes, &mut offset)?;
        let residual_shift = *bytes.get(offset).ok_or(TrainError::InvalidModel(
            "missing block expert residual shift",
        ))?;
        if bytes
            .get(offset + 1..offset + 8)
            .ok_or(TrainError::InvalidModel(
                "missing block expert reserved bytes",
            ))?
            .iter()
            .any(|&value| value != 0)
        {
            return Err(TrainError::InvalidModel("block expert reserved bytes"));
        }
        offset += 8;
        let parameter_count = read_model_usize(bytes, &mut offset)?;
        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || transformer_layers == 0
            || rank == 0
            || rank > MINI_TRANSFORMER_D_MODEL
            || residual_shift > 15
            || parameter_count
                != transformer_layers
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_mul(rank))
                    .ok_or(TrainError::InvalidModel("block expert size overflow"))?
            || bytes.len()
                != HEADER_WITH_CHECKSUM
                    .checked_add(
                        parameter_count
                            .checked_mul(2)
                            .ok_or(TrainError::InvalidModel("block expert payload overflow"))?,
                    )
                    .ok_or(TrainError::InvalidModel("block expert artifact overflow"))?
        {
            return Err(TrainError::InvalidModel("block expert header mismatch"));
        }
        let mut expansion_weights_q15 = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            let end = offset
                .checked_add(2)
                .ok_or(TrainError::InvalidModel("block expert offset overflow"))?;
            let raw: [u8; 2] = bytes
                .get(offset..end)
                .ok_or(TrainError::InvalidModel("truncated block expert"))?
                .try_into()
                .map_err(|_| TrainError::InvalidModel("truncated block expert"))?;
            expansion_weights_q15.push(i16::from_le_bytes(raw));
            offset = end;
        }
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel("block expert payload mismatch"));
        }
        Ok(Self {
            trunk_model_hash,
            transformer_layers,
            rank,
            projection_seed,
            residual_shift,
            expansion_weights_q15,
        })
    }
}

impl MiniTransformerAdamOptimizerState {
    pub fn new_for_model(
        model: &MiniTransformerMlpModel,
        config: IntegerAdamConfig,
    ) -> Result<Self, TrainError> {
        if !config.is_valid() {
            return Err(TrainError::InvalidConfig);
        }
        let parameter_count = model.optimizer_parameter_count()?;
        Ok(Self {
            context_seq_len: model.context_seq_len,
            step: 0,
            bound_model_hash: model.model_hash(),
            config,
            first_moments: vec![0_i64; parameter_count],
            second_moments: vec![0_u64; parameter_count],
            update_residuals: vec![0_i64; parameter_count],
        })
    }

    pub fn validate_for_model(&self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let parameter_count = model.optimizer_parameter_count()?;
        if !self.config.is_valid()
            || self.context_seq_len != model.context_seq_len
            || self.bound_model_hash != model.model_hash()
            || self.first_moments.len() != parameter_count
            || self.second_moments.len() != parameter_count
            || self.update_residuals.len() != parameter_count
        {
            return Err(TrainError::InvalidModel("optimizer state/model mismatch"));
        }
        Ok(())
    }

    pub fn bind_to_model(&mut self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let parameter_count = model.optimizer_parameter_count()?;
        if !self.config.is_valid()
            || self.context_seq_len != model.context_seq_len
            || self.first_moments.len() != parameter_count
            || self.second_moments.len() != parameter_count
            || self.update_residuals.len() != parameter_count
        {
            return Err(TrainError::InvalidModel("optimizer state shape mismatch"));
        }
        self.bound_model_hash = model.model_hash();
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.first_moments.len()
    }

    pub fn state_hash(&self) -> Result<u64, TrainError> {
        let bytes = self.try_to_bytes()?;
        let checksum_offset = bytes
            .len()
            .checked_sub(8)
            .ok_or(TrainError::InvalidModel("optimizer checksum offset"))?;
        let mut offset = checksum_offset;
        read_u64_le(&bytes, &mut offset)
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if !self.config.is_valid()
            || self.first_moments.len() != self.second_moments.len()
            || self.first_moments.len() != self.update_residuals.len()
            || self.context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("invalid optimizer state"));
        }
        let payload_bytes = self
            .parameter_count()
            .checked_mul(24)
            .ok_or(TrainError::InvalidModel("optimizer state size overflow"))?;
        let mut out = Vec::with_capacity(
            80_usize
                .checked_add(payload_bytes)
                .ok_or(TrainError::InvalidModel("optimizer artifact size overflow"))?,
        );
        out.extend_from_slice(MINI_TRANSFORMER_ADAM_STATE_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_D_MODEL, "d_model exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "heads exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HIDDEN_DIM, "hidden_dim exceeds u32")?.to_le_bytes(),
        );
        push_model_usize(&mut out, self.context_seq_len, "context length exceeds u64")?;
        push_model_usize(
            &mut out,
            self.parameter_count(),
            "parameter count exceeds u64",
        )?;
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.bound_model_hash.to_le_bytes());
        out.extend_from_slice(&self.config.learning_rate.to_le_bytes());
        out.push(self.config.step_shift);
        out.push(self.config.beta1_decay_shift);
        out.push(self.config.beta2_decay_shift);
        out.push(0);
        out.extend_from_slice(&self.config.epsilon.to_le_bytes());
        for &value in &self.first_moments {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.second_moments {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.update_residuals {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = hash_u8_slice(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("valid optimizer state should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        const FIXED_BYTES_WITH_CHECKSUM: usize = 80;
        if bytes.len() < FIXED_BYTES_WITH_CHECKSUM {
            return Err(TrainError::InvalidModel("optimizer artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_ADAM_STATE_MAGIC.len()] != MINI_TRANSFORMER_ADAM_STATE_MAGIC {
            return Err(TrainError::InvalidModel("bad optimizer magic"));
        }
        let checksum_offset = bytes
            .len()
            .checked_sub(8)
            .ok_or(TrainError::InvalidModel("optimizer checksum offset"))?;
        let mut checksum_cursor = checksum_offset;
        let expected_checksum = read_u64_le(bytes, &mut checksum_cursor)?;
        if checksum_cursor != bytes.len()
            || hash_u8_slice(&bytes[..checksum_offset]) != expected_checksum
        {
            return Err(TrainError::InvalidModel("optimizer checksum mismatch"));
        }

        let mut offset = MINI_TRANSFORMER_ADAM_STATE_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let context_seq_len = read_model_usize(bytes, &mut offset)?;
        let parameter_count = read_model_usize(bytes, &mut offset)?;
        let step = read_u64_le(bytes, &mut offset)?;
        let bound_model_hash = read_u64_le(bytes, &mut offset)?;
        let learning_rate = read_u32_le(bytes, &mut offset)? as i32;
        let step_shift = *bytes
            .get(offset)
            .ok_or(TrainError::InvalidModel("missing optimizer step shift"))?;
        let beta1_decay_shift = *bytes
            .get(offset + 1)
            .ok_or(TrainError::InvalidModel("missing optimizer beta1 shift"))?;
        let beta2_decay_shift = *bytes
            .get(offset + 2)
            .ok_or(TrainError::InvalidModel("missing optimizer beta2 shift"))?;
        let reserved = *bytes
            .get(offset + 3)
            .ok_or(TrainError::InvalidModel("missing optimizer reserved byte"))?;
        offset += 4;
        let epsilon = read_u64_le(bytes, &mut offset)?;
        let config = IntegerAdamConfig {
            learning_rate,
            step_shift,
            beta1_decay_shift,
            beta2_decay_shift,
            epsilon,
        };
        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
            || reserved != 0
            || !config.is_valid()
        {
            return Err(TrainError::InvalidModel("optimizer header mismatch"));
        }
        let expected_len = FIXED_BYTES_WITH_CHECKSUM
            .checked_add(
                parameter_count
                    .checked_mul(24)
                    .ok_or(TrainError::InvalidModel("optimizer payload overflow"))?,
            )
            .ok_or(TrainError::InvalidModel("optimizer artifact overflow"))?;
        if bytes.len() != expected_len {
            return Err(TrainError::InvalidModel(
                "optimizer artifact length mismatch",
            ));
        }
        let first_moments = read_i64_vec(bytes, &mut offset, parameter_count)?;
        let second_moments = read_u64_vec(bytes, &mut offset, parameter_count)?;
        let update_residuals = read_i64_vec(bytes, &mut offset, parameter_count)?;
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel(
                "optimizer payload length mismatch",
            ));
        }
        Ok(Self {
            context_seq_len,
            step,
            bound_model_hash,
            config,
            first_moments,
            second_moments,
            update_residuals,
        })
    }
}

impl MiniTransformerMlpSwarmWorkerArtifact {
    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.worker_count == 0
            || self.worker.worker_index >= self.worker_count
            || self.model.model_hash() != self.worker.final_model_hash
        {
            return Err(TrainError::InvalidModel("bad swarm worker artifact"));
        }
        let model_bytes = self.model.try_to_bytes()?;
        let mut out = Vec::with_capacity(checked_model_capacity(224, &[model_bytes.len()])?);
        out.extend_from_slice(MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC);
        push_model_usize(&mut out, self.worker_count, "worker count exceeds u64")?;
        push_model_usize(&mut out, self.token_count, "token count exceeds u64")?;
        out.extend_from_slice(&self.token_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            self.base_window_offset,
            "base window offset exceeds u64",
        )?;
        push_model_usize(&mut out, self.base_stride, "base stride exceeds u64")?;
        push_model_optional_usize(
            &mut out,
            self.base_max_windows,
            "base max windows exceeds u64",
        )?;
        out.extend_from_slice(&self.base_model_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            self.worker.worker_index,
            "worker index exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.window_offset,
            "worker window offset exceeds u64",
        )?;
        push_model_usize(&mut out, self.worker.stride, "worker stride exceeds u64")?;
        push_model_optional_usize(
            &mut out,
            self.worker.max_windows,
            "worker max windows exceeds u64",
        )?;
        out.extend_from_slice(&self.worker.window_hash.to_le_bytes());
        push_model_usize(&mut out, self.worker.windows, "worker windows exceeds u64")?;
        push_model_usize(
            &mut out,
            self.worker.examined_windows,
            "worker examined windows exceeds u64",
        )?;
        push_model_usize(&mut out, self.worker.updates, "worker updates exceeds u64")?;
        push_model_usize(
            &mut out,
            self.worker.accepted_batch_count,
            "worker accepted batches exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rejected_batch_count,
            "worker rejected batches exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rollback_count,
            "worker rollbacks exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rejected_window_count,
            "worker rejected windows exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_invalid_forward_count,
            "worker invalid forward count exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.initial_total_error,
            "worker initial total error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_total_error,
            "worker final total error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.initial_probability_error_q15,
            "worker initial probability error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_probability_error_q15,
            "worker final probability error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_accuracy_per_mille,
            "worker final accuracy exceeds u64",
        )?;
        out.extend_from_slice(&self.worker.final_model_hash.to_le_bytes());
        out.extend_from_slice(&self.worker.final_logits_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            model_bytes.len(),
            "worker model bytes exceeds u64",
        )?;
        out.extend_from_slice(&model_bytes);
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer swarm worker artifact should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len() {
            return Err(TrainError::InvalidModel("swarm worker artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len()]
            != MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC
        {
            return Err(TrainError::InvalidModel("bad swarm worker magic"));
        }

        let mut offset = MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len();
        let worker_count = read_model_usize(bytes, &mut offset)?;
        let token_count = read_model_usize(bytes, &mut offset)?;
        let token_hash = read_u64_le(bytes, &mut offset)?;
        let base_window_offset = read_model_usize(bytes, &mut offset)?;
        let base_stride = read_model_usize(bytes, &mut offset)?;
        let base_max_windows = read_model_optional_usize(bytes, &mut offset)?;
        let base_model_hash = read_u64_le(bytes, &mut offset)?;
        let worker_index = read_model_usize(bytes, &mut offset)?;
        let window_offset = read_model_usize(bytes, &mut offset)?;
        let stride = read_model_usize(bytes, &mut offset)?;
        let max_windows = read_model_optional_usize(bytes, &mut offset)?;
        let window_hash = read_u64_le(bytes, &mut offset)?;
        let windows = read_model_usize(bytes, &mut offset)?;
        let examined_windows = read_model_usize(bytes, &mut offset)?;
        let updates = read_model_usize(bytes, &mut offset)?;
        let accepted_batch_count = read_model_usize(bytes, &mut offset)?;
        let rejected_batch_count = read_model_usize(bytes, &mut offset)?;
        let rollback_count = read_model_usize(bytes, &mut offset)?;
        let rejected_window_count = read_model_usize(bytes, &mut offset)?;
        let final_invalid_forward_count = read_model_usize(bytes, &mut offset)?;
        let initial_total_error = read_model_usize(bytes, &mut offset)?;
        let final_total_error = read_model_usize(bytes, &mut offset)?;
        let initial_probability_error_q15 = read_model_usize(bytes, &mut offset)?;
        let final_probability_error_q15 = read_model_usize(bytes, &mut offset)?;
        let final_accuracy_per_mille = read_model_usize(bytes, &mut offset)?;
        let final_model_hash = read_u64_le(bytes, &mut offset)?;
        let final_logits_hash = read_u64_le(bytes, &mut offset)?;
        let model_len = read_model_usize(bytes, &mut offset)?;
        let model_end = offset
            .checked_add(model_len)
            .ok_or(TrainError::InvalidModel(
                "swarm worker model offset overflow",
            ))?;
        let model_bytes = bytes
            .get(offset..model_end)
            .ok_or(TrainError::InvalidModel("swarm worker model truncated"))?;
        offset = model_end;
        if offset != bytes.len() || worker_count == 0 || worker_index >= worker_count {
            return Err(TrainError::InvalidModel("bad swarm worker header"));
        }

        let model = MiniTransformerMlpModel::from_bytes(model_bytes)?;
        if model.model_hash() != final_model_hash {
            return Err(TrainError::InvalidModel("swarm worker model hash mismatch"));
        }
        Ok(Self {
            worker_count,
            token_count,
            token_hash,
            base_window_offset,
            base_stride,
            base_max_windows,
            base_model_hash,
            worker: MiniTransformerMlpSwarmWorkerTrace {
                worker_index,
                window_offset,
                stride,
                max_windows,
                token_hash,
                window_hash,
                windows,
                examined_windows,
                updates,
                accepted_batch_count,
                rejected_batch_count,
                rollback_count,
                rejected_window_count,
                final_invalid_forward_count,
                initial_total_error,
                final_total_error,
                initial_probability_error_q15,
                final_probability_error_q15,
                final_accuracy_per_mille,
                final_model_hash,
                final_logits_hash,
            },
            model,
        })
    }

    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_WORKER_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_worker",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "base_max_windows", self.base_max_windows);
        comma(&mut out);
        push_hash_field(&mut out, "base_model_hash", self.base_model_hash);
        out.push('}');
        comma(&mut out);
        push_quoted(&mut out, "worker");
        out.push(':');
        push_mini_transformer_swarm_worker(&mut out, &self.worker);
        comma(&mut out);
        out.push_str("\"artifact\":{");
        push_string_field(&mut out, "format", "nsrlswarm-worker");
        comma(&mut out);
        push_string_field(&mut out, "magic", "NSRLWK1");
        comma(&mut out);
        push_usize_field(
            &mut out,
            "model_bytes",
            self.model
                .try_to_bytes()
                .map(|bytes| bytes.len())
                .unwrap_or(0),
        );
        out.push('}');
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmModel {
    pub fn new(
        best_worker_index: usize,
        workers: Vec<MiniTransformerMlpModel>,
    ) -> Result<Self, TrainError> {
        let first = workers
            .first()
            .ok_or(TrainError::InvalidModel("empty mini transformer swarm"))?;
        if best_worker_index >= workers.len() {
            return Err(TrainError::InvalidModel("swarm best worker out of range"));
        }
        let context_seq_len = first.context_seq_len;
        if context_seq_len == 0
            || workers
                .iter()
                .any(|worker| worker.context_seq_len != context_seq_len)
        {
            return Err(TrainError::InvalidModel(
                "swarm worker context length mismatch",
            ));
        }
        Ok(Self {
            context_seq_len,
            best_worker_index,
            workers,
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.context_seq_len);
        hasher.update_usize(self.best_worker_index);
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.model_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn embedding_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.embedding_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn attention_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.attention_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn mlp_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.mlp_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn output_head_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.output_head_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        let worker_blobs = self
            .workers
            .iter()
            .map(MiniTransformerMlpModel::try_to_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let payload_bytes = worker_blobs.iter().try_fold(0_usize, |total, blob| {
            total
                .checked_add(8)
                .and_then(|value| value.checked_add(blob.len()))
                .ok_or(TrainError::InvalidModel("swarm artifact length overflow"))
        })?;
        let mut out = Vec::with_capacity(checked_model_capacity(32, &[payload_bytes])?);
        out.extend_from_slice(MINI_TRANSFORMER_SWARM_MODEL_MAGIC);
        out.extend_from_slice(
            &checked_u32(
                self.context_seq_len,
                "mini transformer swarm context_seq_len exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.workers.len(),
                "mini transformer swarm worker count exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.best_worker_index,
                "mini transformer swarm best worker exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&self.model_hash().to_le_bytes());
        for blob in worker_blobs {
            out.extend_from_slice(
                &checked_u64(blob.len(), "mini transformer worker blob exceeds u64")?.to_le_bytes(),
            );
            out.extend_from_slice(&blob);
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer swarm model should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len() + 4 + 4 + 4 + 4 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("swarm artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len()] != MINI_TRANSFORMER_SWARM_MODEL_MAGIC
        {
            return Err(TrainError::InvalidModel("bad swarm magic"));
        }
        let mut offset = MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len();
        let context_seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let worker_count = read_u32_le(bytes, &mut offset)? as usize;
        let best_worker_index = read_u32_le(bytes, &mut offset)? as usize;
        let reserved = read_u32_le(bytes, &mut offset)?;
        let expected_model_hash = read_u64_le(bytes, &mut offset)?;
        if worker_count == 0 || best_worker_index >= worker_count || reserved != 0 {
            return Err(TrainError::InvalidModel("bad swarm header"));
        }

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let blob_len = read_u64_le(bytes, &mut offset)? as usize;
            let blob_end = offset
                .checked_add(blob_len)
                .ok_or(TrainError::InvalidModel("swarm worker offset overflow"))?;
            let blob = bytes
                .get(offset..blob_end)
                .ok_or(TrainError::InvalidModel("swarm worker blob truncated"))?;
            workers.push(MiniTransformerMlpModel::from_bytes(blob)?);
            offset = blob_end;
        }
        if offset != bytes.len() {
            return Err(TrainError::InvalidModel("swarm artifact length mismatch"));
        }

        let model = Self::new(best_worker_index, workers)?;
        if model.context_seq_len != context_seq_len {
            return Err(TrainError::InvalidModel("swarm context hash mismatch"));
        }
        if model.model_hash() != expected_model_hash {
            return Err(TrainError::InvalidModel("swarm model hash mismatch"));
        }
        Ok(model)
    }

    pub fn to_expert_manifest(&self) -> Result<MiniTransformerMlpSwarmExpertManifest, TrainError> {
        Ok(MiniTransformerMlpSwarmExpertManifest {
            artifact_format: "nsrlswarm",
            artifact_magic: "NSRLSW1",
            artifact_byte_count: self.try_to_bytes()?.len(),
            model_id: MINI_TRANSFORMER_SWARM_MODEL_ID,
            tokenizer: BYTE_TOKENIZER_ID,
            context_seq_len: self.context_seq_len,
            worker_count: self.worker_count(),
            best_worker_index: self.best_worker_index,
            parameter_bytes: self.parameter_bytes(),
            model_hash: self.model_hash(),
            embedding_hash: self.embedding_hash(),
            attention_hash: self.attention_hash(),
            mlp_hash: self.mlp_hash(),
            output_head_hash: self.output_head_hash(),
            worker_model_hashes: self
                .workers
                .iter()
                .map(MiniTransformerMlpModel::model_hash)
                .collect(),
            worker_parameter_bytes: self
                .workers
                .iter()
                .map(mini_transformer_mlp_parameter_bytes)
                .collect(),
        })
    }

    pub fn parameter_bytes(&self) -> usize {
        self.workers
            .iter()
            .map(mini_transformer_mlp_parameter_bytes)
            .fold(0_usize, usize::saturating_add)
    }
}

fn mini_transformer_mlp_parameter_bytes(model: &MiniTransformerMlpModel) -> usize {
    model
        .embeddings
        .len()
        .saturating_add(model.position_embeddings.len())
        .saturating_mul(core::mem::size_of::<i16>())
        .saturating_add(model.q_weights.len())
        .saturating_add(model.k_weights.len())
        .saturating_add(model.v_weights.len())
        .saturating_add(model.o_weights.len())
        .saturating_add(model.up_weights.len())
        .saturating_add(model.gate_weights.len())
        .saturating_add(model.down_weights.len())
        .saturating_add(model.output_weights.len())
}

impl MiniTransformerMlpSwarmExpertManifest {
    pub fn capability_tags(&self) -> &'static [&'static str] {
        MINI_TRANSFORMER_SWARM_CAPABILITY_TAGS
    }

    pub fn supports_capability(&self, capability: &str) -> bool {
        self.capability_tags().contains(&capability)
    }

    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            MINI_TRANSFORMER_SWARM_EXPERT_MANIFEST_SCHEMA,
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", self.model_id);
        comma(&mut out);
        out.push_str("\"artifact\":{");
        push_string_field(&mut out, "format", self.artifact_format);
        comma(&mut out);
        push_string_field(&mut out, "magic", self.artifact_magic);
        comma(&mut out);
        push_usize_field(&mut out, "bytes", self.artifact_byte_count);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"tokenizer\":{");
        push_string_field(&mut out, "id", self.tokenizer);
        comma(&mut out);
        push_string_field(&mut out, "contract", "identity_u8_bytes");
        out.push('}');
        comma(&mut out);
        out.push_str("\"interfaces\":{");
        push_string_field(&mut out, "input_schema", "nsrl.byte_prompt.v1");
        comma(&mut out);
        push_string_field(&mut out, "output_schema", "nsrl.byte_generation.v1");
        comma(&mut out);
        push_string_field(
            &mut out,
            "generation_trace_schema",
            MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"numeric_contract\":{");
        push_string_field(&mut out, "residual_scale", "q15_i16");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "qint8");
        comma(&mut out);
        push_string_field(&mut out, "activation_dtype", "qint16");
        comma(&mut out);
        push_string_field(&mut out, "accumulator_dtype", "qint64");
        comma(&mut out);
        push_string_field(&mut out, "softmax", "base2_q15");
        out.push('}');
        comma(&mut out);
        out.push_str("\"model_shape\":{");
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        out.push('}');
        comma(&mut out);
        out.push_str("\"hashes\":{");
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_hash_array_field(&mut out, "worker_model_hashes", &self.worker_model_hashes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"capabilities\":{");
        push_string_array_field(&mut out, "tags", self.capability_tags());
        out.push('}');
        comma(&mut out);
        out.push_str("\"routing_hints\":{");
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_string_field(&mut out, "default_composition", "average_logits");
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "supported_compositions",
            &["average_logits", "confidence_weighted", "confidence_router"],
        );
        comma(&mut out);
        push_string_field(&mut out, "confidence_signal", "top_logit_margin_q8");
        comma(&mut out);
        push_string_field(&mut out, "tie_breaker", "lowest_worker_index");
        out.push('}');
        comma(&mut out);
        out.push_str("\"budgets\":{");
        push_usize_field(&mut out, "artifact_bytes", self.artifact_byte_count);
        comma(&mut out);
        push_usize_field(&mut out, "parameter_bytes", self.parameter_bytes);
        comma(&mut out);
        push_usize_array_field(
            &mut out,
            "worker_parameter_bytes",
            &self.worker_parameter_bytes,
        );
        comma(&mut out);
        push_bool_field(&mut out, "wasm_bundle_budget_known", false);
        out.push('}');
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "not_a_general_purpose_language_model",
                "byte_level_contract_only",
                "single_block_mini_transformer_workers",
                "router_hints_are_symbolic_not_learned",
                "wasm_bundle_budget_not_measured_yet",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}

pub fn route_mini_transformer_swarm_experts(
    candidates: &[MiniTransformerSwarmRouteCandidate],
    config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    route_mini_transformer_swarm_experts_with_prompt_affinity(candidates, config, prompt, None)
}

pub fn route_mini_transformer_swarm_expert_models(
    experts: &[MiniTransformerSwarmRoutedGenerationExpert],
    route_config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    let candidates = experts
        .iter()
        .map(|expert| {
            Ok(MiniTransformerSwarmRouteCandidate {
                expert_id: expert.expert_id.clone(),
                manifest: expert.model.to_expert_manifest()?,
            })
        })
        .collect::<Result<Vec<_>, TrainError>>()?;
    let prompt_affinities = if route_config.prompt_affinity {
        Some(
            experts
                .iter()
                .map(|expert| {
                    mini_transformer_swarm_prompt_affinity(
                        &expert.model,
                        prompt,
                        attention_kind,
                        position_policy,
                        composition,
                        route_config.prompt_affinity_max_windows,
                    )
                })
                .collect::<Result<Vec<_>, TrainError>>()?,
        )
    } else {
        None
    };
    route_mini_transformer_swarm_experts_with_prompt_affinity(
        &candidates,
        route_config,
        prompt,
        prompt_affinities.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_routed_mini_transformer_swarm_experts(
    experts: &[MiniTransformerSwarmRoutedGenerationExpert],
    route_config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmRoutedGenerationTrace, TrainError> {
    let route = route_mini_transformer_swarm_expert_models(
        experts,
        route_config,
        prompt,
        attention_kind,
        position_policy,
        composition,
    )?;
    let mut selected_expert_ids = Vec::with_capacity(route.selected_expert_indices.len());
    let mut active_workers = Vec::new();
    let mut best_worker_index = None;

    for &expert_index in &route.selected_expert_indices {
        let expert = experts.get(expert_index).ok_or(TrainError::InvalidConfig)?;
        selected_expert_ids.push(expert.expert_id.clone());
        let worker_offset = active_workers.len();
        if best_worker_index.is_none() {
            best_worker_index = Some(worker_offset.saturating_add(expert.model.best_worker_index));
        }
        active_workers.extend(expert.model.workers.iter().cloned());
    }

    let active_model =
        MiniTransformerMlpSwarmModel::new(best_worker_index.unwrap_or(0), active_workers)?;
    let generation =
        generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
            &active_model,
            prompt,
            config,
            attention_kind,
            position_policy,
            composition,
            decode_priors,
        )?;

    Ok(MiniTransformerSwarmRoutedGenerationTrace {
        route,
        selected_expert_ids,
        active_worker_count: active_model.worker_count(),
        generation,
    })
}

fn route_mini_transformer_swarm_experts_with_prompt_affinity(
    candidates: &[MiniTransformerSwarmRouteCandidate],
    config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    prompt_affinities: Option<&[MiniTransformerSwarmPromptAffinityTrace]>,
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    if candidates.is_empty() || config.active_expert_limit == 0 {
        return Err(TrainError::InvalidConfig);
    }
    if let Some(affinities) = prompt_affinities
        && affinities.len() != candidates.len()
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut candidate_traces = Vec::with_capacity(candidates.len());
    for (expert_index, candidate) in candidates.iter().enumerate() {
        candidate_traces.push(mini_transformer_swarm_route_candidate_trace(
            expert_index,
            candidate,
            &config,
            prompt_affinities.and_then(|affinities| affinities.get(expert_index)),
        ));
    }

    let mut selected = candidate_traces
        .iter()
        .filter(|candidate| candidate.accepted)
        .collect::<Vec<_>>();
    selected.sort_by_key(|candidate| {
        (
            core::cmp::Reverse(candidate.score),
            candidate.parameter_bytes,
            candidate.artifact_bytes,
            candidate.expert_index,
        )
    });
    let selected_expert_indices = selected
        .into_iter()
        .take(config.active_expert_limit)
        .map(|candidate| candidate.expert_index)
        .collect::<Vec<_>>();
    if selected_expert_indices.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    Ok(MiniTransformerSwarmRouteDecisionTrace {
        config,
        prompt_bytes: prompt.to_vec(),
        selected_expert_indices,
        candidates: candidate_traces,
    })
}

fn mini_transformer_swarm_route_candidate_trace(
    expert_index: usize,
    candidate: &MiniTransformerSwarmRouteCandidate,
    config: &MiniTransformerSwarmRouteConfig,
    prompt_affinity: Option<&MiniTransformerSwarmPromptAffinityTrace>,
) -> MiniTransformerSwarmRouteCandidateTrace {
    let (capability_match, matched_capabilities, missing_capabilities) =
        mini_transformer_swarm_route_capability_match(
            &candidate.manifest,
            &config.required_capabilities,
        );
    let reject_reason = if !capability_match {
        "capability_mismatch"
    } else if config
        .max_artifact_bytes
        .is_some_and(|max| candidate.manifest.artifact_byte_count > max)
    {
        "artifact_budget_exceeded"
    } else if config
        .max_parameter_bytes
        .is_some_and(|max| candidate.manifest.parameter_bytes > max)
    {
        "parameter_budget_exceeded"
    } else {
        ""
    };
    let accepted = reject_reason.is_empty();
    let manifest_score = if accepted {
        mini_transformer_swarm_route_score(&candidate.manifest, capability_match)
    } else {
        0
    };
    let prompt_affinity_score = if accepted {
        prompt_affinity.map(|affinity| affinity.score).unwrap_or(0)
    } else {
        0
    };
    let score = manifest_score.saturating_add(prompt_affinity_score);

    MiniTransformerSwarmRouteCandidateTrace {
        expert_index,
        expert_id: candidate.expert_id.clone(),
        accepted,
        reject_reason,
        score,
        manifest_score,
        prompt_affinity_score,
        prompt_eval_windows: prompt_affinity
            .map(|affinity| affinity.eval_windows)
            .unwrap_or(0),
        prompt_probability_error_q15: prompt_affinity
            .map(|affinity| affinity.probability_error_q15),
        capability_match,
        matched_capabilities,
        missing_capabilities,
        model_hash: candidate.manifest.model_hash,
        artifact_bytes: candidate.manifest.artifact_byte_count,
        parameter_bytes: candidate.manifest.parameter_bytes,
        worker_count: candidate.manifest.worker_count,
        context_seq_len: candidate.manifest.context_seq_len,
        default_composition: "average_logits",
    }
}

fn mini_transformer_swarm_route_capability_match(
    manifest: &MiniTransformerMlpSwarmExpertManifest,
    required_capabilities: &[String],
) -> (bool, Vec<String>, Vec<String>) {
    let mut matched = Vec::new();
    let mut missing = Vec::new();
    for capability in required_capabilities {
        if matched.iter().any(|seen| seen == capability)
            || missing.iter().any(|seen| seen == capability)
        {
            continue;
        }
        if manifest.supports_capability(capability) {
            matched.push(capability.clone());
        } else {
            missing.push(capability.clone());
        }
    }
    (missing.is_empty(), matched, missing)
}

fn mini_transformer_swarm_route_score(
    manifest: &MiniTransformerMlpSwarmExpertManifest,
    capability_match: bool,
) -> i64 {
    let capability_score = if capability_match { 1_000_000_i64 } else { 0 };
    let worker_score = i64::try_from(manifest.worker_count.min(4096)).unwrap_or(i64::MAX) * 1_000;
    let context_score = i64::try_from(manifest.context_seq_len.min(4096)).unwrap_or(i64::MAX);
    let budget_penalty =
        i64::try_from((manifest.parameter_bytes / 4096).min(i64::MAX as usize)).unwrap_or(i64::MAX);
    capability_score
        .saturating_add(worker_score)
        .saturating_add(context_score)
        .saturating_sub(budget_penalty)
}

impl MiniTransformerSwarmRouteDecisionTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_ROUTE_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_mini_transformer_swarm_route_config_field(&mut out, "config", &self.config);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_hash_field(&mut out, "hash", hash_u8_slice(&self.prompt_bytes));
        out.push('}');
        comma(&mut out);
        push_usize_array_field(
            &mut out,
            "selected_expert_indices",
            &self.selected_expert_indices,
        );
        comma(&mut out);
        push_mini_transformer_swarm_route_candidates_field(
            &mut out,
            "candidates",
            &self.candidates,
        );
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "deterministic_router_not_trained_router_weights",
                "prompt_affinity_is_fixed_prompt_replay_when_enabled",
                "does_not_run_generation",
                "does_not_measure_request_latency_yet",
                "does_not_rank_semantic_quality",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerSwarmRoutedGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            MINI_TRANSFORMER_SWARM_ROUTED_GENERATION_SCHEMA,
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "selected_expert_ids",
            &self
                .selected_expert_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        comma(&mut out);
        push_usize_field(&mut out, "active_worker_count", self.active_worker_count);
        comma(&mut out);
        push_json_line_object_field(&mut out, "route", &self.route.to_json_line());
        comma(&mut out);
        push_json_line_object_field(&mut out, "generation", &self.generation.to_json_line());
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "routes_by_manifest_and_optional_prompt_affinity_not_trained_semantic_router",
                "active_set_workers_are_concatenated_before_generation",
                "does_not_train_router_weights_yet",
                "does_not_claim_language_model_quality",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", MINI_TRANSFORMER_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_string_field(&mut out, "attention_kind", self.attention_kind.as_str());
        comma(&mut out);
        push_string_field(&mut out, "position_policy", self.position_policy.as_str());
        comma(&mut out);
        push_bool_field(
            &mut out,
            "incremental_attention_state",
            self.attention_kind.uses_incremental_state(),
        );
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.prompt_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"generation\":{");
        push_usize_field(&mut out, "new_tokens", self.generated_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.generated_bytes);
        out.push('}');
        comma(&mut out);
        push_mini_transformer_ttt_stats_field(&mut out, "ttt", self.ttt_stats);
        comma(&mut out);
        push_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        let known_non_claims: &[&str] = if self.attention_kind.uses_incremental_state() {
            &MINI_TRANSFORMER_STREAMING_GENERATION_KNOWN_NON_CLAIMS
        } else if self.position_policy == MiniTransformerPositionPolicy::Nope {
            &MINI_TRANSFORMER_NOPE_GENERATION_KNOWN_NON_CLAIMS
        } else {
            &MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS
        };
        push_string_array_field(&mut out, "known_non_claims", known_non_claims);
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerSwarmGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", MINI_TRANSFORMER_SWARM_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_string_field(&mut out, "attention_kind", self.attention_kind.as_str());
        comma(&mut out);
        push_string_field(&mut out, "position_policy", self.position_policy.as_str());
        comma(&mut out);
        push_string_field(&mut out, "composition", self.composition.as_str());
        comma(&mut out);
        push_bool_field(
            &mut out,
            "incremental_attention_state",
            self.attention_kind.uses_incremental_state(),
        );
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
        comma(&mut out);
        push_hash_field(&mut out, "swarm_model_hash", self.swarm_model_hash);
        comma(&mut out);
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.prompt_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"generation\":{");
        push_usize_field(&mut out, "new_tokens", self.generated_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.generated_bytes);
        out.push('}');
        comma(&mut out);
        push_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        let known_non_claims: &[&str] =
            if self.position_policy == MiniTransformerPositionPolicy::Nope {
                &MINI_TRANSFORMER_NOPE_GENERATION_KNOWN_NON_CLAIMS
            } else {
                &MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS
            };
        push_string_array_field(&mut out, "known_non_claims", known_non_claims);
        out.push('}');
        out.push('\n');
        out
    }
}

pub fn generate_mini_transformer(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_priors(model, prompt, config, None)
}

pub fn generate_mini_transformer_swarm(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
        model,
        prompt,
        config,
        MiniTransformerAttentionKind::Linear,
        MiniTransformerPositionPolicy::Nope,
        MiniTransformerSwarmComposition::AverageLogits,
        None,
    )
}

pub fn generate_mini_transformer_swarm_with_attention_kind_position_policy_and_priors(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        position_policy,
        MiniTransformerSwarmComposition::AverageLogits,
        decode_priors,
    )
}

pub fn generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    if prompt.is_empty()
        || model.context_seq_len == 0
        || model.workers.is_empty()
        || attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let row = mini_transformer_swarm_ensemble_row_for_context(
            model,
            context_window,
            attention_kind,
            position_policy,
            composition,
        )?;
        let selection = select_byte_from_row_with_priors(
            &row.logits_q8,
            &row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: row.logits_q8[predicted_index],
            predicted_probability_q15: row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });
    }

    Ok(MiniTransformerSwarmGenerationTrace {
        config,
        attention_kind,
        position_policy,
        composition,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        swarm_model_hash: model.model_hash(),
        worker_count: model.worker_count(),
        best_worker_index: model.best_worker_index,
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        steps,
    })
}

fn mini_transformer_swarm_ensemble_row_for_context(
    model: &MiniTransformerMlpSwarmModel,
    context_window: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
) -> Result<ByteVocabOutputRow, TrainError> {
    if model.workers.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut worker_rows = Vec::with_capacity(model.workers.len());
    for worker in &model.workers {
        let cache = mini_transformer_forward_for_attention_and_position(
            worker,
            context_window,
            attention_kind,
            position_policy,
        )?;
        worker_rows.push(cache);
    }

    let logits_q8 = match composition {
        MiniTransformerSwarmComposition::AverageLogits => {
            mini_transformer_average_worker_logits_q8(&worker_rows)
        }
        MiniTransformerSwarmComposition::ConfidenceWeighted => {
            mini_transformer_confidence_weighted_worker_logits_q8(&worker_rows)
        }
        MiniTransformerSwarmComposition::ConfidenceRouter => {
            mini_transformer_confidence_routed_worker_logits_q8(&worker_rows)
        }
    };
    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15).ok_or(TrainError::CoreRejected(
        "mini_transformer_swarm_output_softmax",
    ))?;

    Ok(ByteVocabOutputRow {
        logits_q8,
        probabilities_q15,
    })
}

fn mini_transformer_swarm_prompt_affinity(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    max_windows: usize,
) -> Result<MiniTransformerSwarmPromptAffinityTrace, TrainError> {
    if prompt.len() < 2 || max_windows == 0 {
        return Ok(MiniTransformerSwarmPromptAffinityTrace {
            eval_windows: 0,
            probability_error_q15: 0,
            score: 0,
        });
    }
    if model.context_seq_len == 0
        || model.workers.is_empty()
        || attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }

    let start = prompt.len().saturating_sub(max_windows.saturating_add(1));
    let mut eval_windows = 0_usize;
    let mut probability_error_q15 = 0_usize;
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for target_index in start.max(1)..prompt.len() {
        let context = &prompt[..target_index];
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let row = mini_transformer_swarm_ensemble_row_for_context(
            model,
            context_window,
            attention_kind,
            position_policy,
            composition,
        )?;
        probability_error_q15 = probability_error_q15.saturating_add(
            byte_sample_probability_error_q15(&row.probabilities_q15, prompt[target_index]),
        );
        eval_windows = eval_windows.saturating_add(1);
    }

    let mean_error = probability_error_q15
        .checked_div(eval_windows)
        .and_then(|value| i64::try_from(value).ok())
        .unwrap_or(i64::from(i16::MAX));
    let score = i64::from(i16::MAX)
        .saturating_sub(mean_error)
        .saturating_mul(10);

    Ok(MiniTransformerSwarmPromptAffinityTrace {
        eval_windows,
        probability_error_q15,
        score,
    })
}

fn mini_transformer_average_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    let mut sums = [0_i64; BYTE_VOCAB];
    for row in rows {
        for (sum, &logit) in sums.iter_mut().zip(row.logits_q8.iter()) {
            *sum = sum.saturating_add(i64::from(logit));
        }
    }
    let divisor = rows.len().max(1) as i64;
    mini_transformer_logit_sums_to_q8(&sums, divisor)
}

fn mini_transformer_confidence_weighted_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    let mut sums = [0_i64; BYTE_VOCAB];
    let mut total_weight = 0_i64;
    for row in rows {
        let weight = i64::from(mini_transformer_logit_margin_q8(&row.logits_q8).max(1));
        total_weight = total_weight.saturating_add(weight);
        for (sum, &logit) in sums.iter_mut().zip(row.logits_q8.iter()) {
            *sum = sum.saturating_add(i64::from(logit).saturating_mul(weight));
        }
    }
    mini_transformer_logit_sums_to_q8(&sums, total_weight.max(1))
}

fn mini_transformer_confidence_routed_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    rows.iter()
        .enumerate()
        .max_by_key(|&(index, row)| {
            (
                mini_transformer_logit_margin_q8(&row.logits_q8),
                core::cmp::Reverse(index),
            )
        })
        .map(|(_, row)| row.logits_q8)
        .unwrap_or([0_i32; BYTE_VOCAB])
}

fn mini_transformer_logit_sums_to_q8(sums: &[i64; BYTE_VOCAB], divisor: i64) -> [i32; BYTE_VOCAB] {
    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (out, &sum) in logits_q8.iter_mut().zip(sums.iter()) {
        let averaged = sum / divisor.max(1);
        *out = averaged.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    logits_q8
}

fn mini_transformer_logit_margin_q8(logits_q8: &[i32; BYTE_VOCAB]) -> i32 {
    let mut best = i32::MIN;
    let mut second = i32::MIN;
    for &logit in logits_q8 {
        if logit > best {
            second = best;
            best = logit;
        } else if logit > second {
            second = logit;
        }
    }
    best.saturating_sub(second).max(0)
}

pub fn generate_mini_transformer_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_and_priors(
        model,
        prompt,
        config,
        MiniTransformerAttentionKind::Base2Softmax,
        decode_priors,
    )
}

pub fn generate_mini_transformer_with_attention_kind(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        None,
    )
}

pub fn mini_transformer_next_token_row_with_attention_kind_position_policy(
    model: &MiniTransformerMlpModel,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let attention_kind = if model.transformer_layers() == 1 {
        attention_kind.preferred_generation_kind(position_policy)
    } else {
        attention_kind
    };
    if attention_kind.uses_incremental_state() {
        return Err(TrainError::InvalidConfig);
    }
    let cache = mini_transformer_forward_for_attention_and_position(
        model,
        context,
        attention_kind,
        position_policy,
    )?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: cache.logits_q8,
        probabilities_q15: cache.probabilities_q15,
    })
}

pub fn generate_mini_transformer_with_attention_kind_and_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_position_policy_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        MiniTransformerPositionPolicy::LearnedAbsolute,
        decode_priors,
    )
}

pub fn generate_mini_transformer_with_attention_kind_position_policy_and_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
        model,
        prompt,
        config,
        attention_kind,
        position_policy,
        decode_priors,
        DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT,
    )
}

pub fn generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
    ttt_learning_rate_shift: u8,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    let attention_kind = if model.transformer_layers() == 1 {
        attention_kind.preferred_generation_kind(position_policy)
    } else {
        attention_kind
    };
    if attention_kind == MiniTransformerAttentionKind::LinearStreamingNope {
        return generate_mini_transformer_streaming_linear_nope_with_priors(
            model,
            prompt,
            config,
            decode_priors,
        );
    }
    if attention_kind == MiniTransformerAttentionKind::LinearStreamingTttNope {
        return generate_mini_transformer_streaming_linear_ttt_nope_with_priors(
            model,
            prompt,
            config,
            decode_priors,
            ttt_learning_rate_shift,
        );
    }

    if prompt.is_empty() || model.context_seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let cache = mini_transformer_forward_for_attention_and_position(
            model,
            context_window,
            attention_kind,
            position_policy,
        )?;
        let selection = select_byte_from_row_with_priors(
            &cache.logits_q8,
            &cache.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: cache.logits_q8[predicted_index],
            predicted_probability_q15: cache.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind,
        position_policy,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: None,
        steps,
    })
}

fn generate_mini_transformer_streaming_linear_nope_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    if prompt.is_empty() || model.context_seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut workspace = MiniTransformerStreamingLinearWorkspace::new()?;
    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    let mut current_row = None;
    for &token in prompt {
        current_row = Some(mini_transformer_streaming_linear_nope_step(
            model,
            token,
            &mut workspace,
        )?);
    }
    let mut current_row = current_row.ok_or(TrainError::InvalidConfig)?;

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let selection = select_byte_from_row_with_priors(
            &current_row.logits_q8,
            &current_row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: current_row.logits_q8[predicted_index],
            predicted_probability_q15: current_row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });

        if step_index + 1 < config.max_new_tokens {
            current_row = mini_transformer_streaming_linear_nope_step(
                model,
                predicted_token,
                &mut workspace,
            )?;
        }
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind: MiniTransformerAttentionKind::LinearStreamingNope,
        position_policy: MiniTransformerPositionPolicy::Nope,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: None,
        steps,
    })
}

fn generate_mini_transformer_streaming_linear_ttt_nope_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
    ttt_learning_rate_shift: u8,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    if prompt.is_empty() || model.context_seq_len == 0 || ttt_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut workspace = MiniTransformerStreamingLinearWorkspace::new()?;
    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut prompt_state_delta_l1 = 0_u64;
    let mut generated_state_delta_l1 = 0_u64;
    let mut zero_delta_count = 0_usize;
    let mut step_count = 0_usize;

    let mut current_row = None;
    for &token in prompt {
        let (row, delta_l1) = mini_transformer_streaming_linear_ttt_nope_step(
            model,
            token,
            &mut workspace,
            ttt_learning_rate_shift,
        )?;
        current_row = Some(row);
        prompt_state_delta_l1 = prompt_state_delta_l1.saturating_add(delta_l1);
        step_count = step_count.saturating_add(1);
        if delta_l1 == 0 {
            zero_delta_count = zero_delta_count.saturating_add(1);
        }
    }
    let mut current_row = current_row.ok_or(TrainError::InvalidConfig)?;

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let selection = select_byte_from_row_with_priors(
            &current_row.logits_q8,
            &current_row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: current_row.logits_q8[predicted_index],
            predicted_probability_q15: current_row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });

        let (row, delta_l1) = mini_transformer_streaming_linear_ttt_nope_step(
            model,
            predicted_token,
            &mut workspace,
            ttt_learning_rate_shift,
        )?;
        current_row = row;
        generated_state_delta_l1 = generated_state_delta_l1.saturating_add(delta_l1);
        step_count = step_count.saturating_add(1);
        if delta_l1 == 0 {
            zero_delta_count = zero_delta_count.saturating_add(1);
        }
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind: MiniTransformerAttentionKind::LinearStreamingTttNope,
        position_policy: MiniTransformerPositionPolicy::Nope,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: Some(MiniTransformerStreamingTttStats {
            learning_rate_shift: ttt_learning_rate_shift,
            step_count,
            zero_delta_count,
            prompt_state_delta_l1,
            generated_state_delta_l1,
            total_state_delta_l1: prompt_state_delta_l1.saturating_add(generated_state_delta_l1),
        }),
        steps,
    })
}

struct MiniTransformerStreamingLinearWorkspace {
    attention_q: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_k: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_v: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_context: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_prediction: [i16; MINI_TRANSFORMER_D_MODEL],
    state_kv: Vec<i64>,
    key_sums: Vec<i64>,
    embedding_output: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_output: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_residual: [i16; MINI_TRANSFORMER_D_MODEL],
    mlp_up: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_gate: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_gated: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_output: [i16; MINI_TRANSFORMER_D_MODEL],
    block_output: [i16; MINI_TRANSFORMER_D_MODEL],
}

impl MiniTransformerStreamingLinearWorkspace {
    fn new() -> Result<Self, TrainError> {
        let (state_len, key_sum_len) =
            linear_attention_state_lengths(MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HEADS)
                .ok_or(TrainError::InvalidConfig)?;
        let mut workspace = Self {
            attention_q: [0; MINI_TRANSFORMER_D_MODEL],
            attention_k: [0; MINI_TRANSFORMER_D_MODEL],
            attention_v: [0; MINI_TRANSFORMER_D_MODEL],
            attention_context: [0; MINI_TRANSFORMER_D_MODEL],
            attention_prediction: [0; MINI_TRANSFORMER_D_MODEL],
            state_kv: vec![0; state_len],
            key_sums: vec![0; key_sum_len],
            embedding_output: [0; MINI_TRANSFORMER_D_MODEL],
            attention_output: [0; MINI_TRANSFORMER_D_MODEL],
            attention_residual: [0; MINI_TRANSFORMER_D_MODEL],
            mlp_up: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_gate: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_gated: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_output: [0; MINI_TRANSFORMER_D_MODEL],
            block_output: [0; MINI_TRANSFORMER_D_MODEL],
        };
        clear_linear_attention_state_checked(
            MINI_TRANSFORMER_D_MODEL,
            MINI_TRANSFORMER_HEADS,
            LinearAttentionState {
                state_kv: &mut workspace.state_kv,
                key_sums: &mut workspace.key_sums,
            },
        )
        .ok_or(TrainError::InvalidConfig)?;
        Ok(workspace)
    }
}

fn mini_transformer_streaming_linear_nope_step(
    model: &MiniTransformerMlpModel,
    token: u8,
    workspace: &mut MiniTransformerStreamingLinearWorkspace,
) -> Result<ByteVocabOutputRow, TrainError> {
    mini_transformer_embedding_token_nope_q15(
        &model.embeddings,
        token,
        &mut workspace.embedding_output,
    )?;

    let attention_params = SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: &model.q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: &model.k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: &model.v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: &model.o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    };
    linear_attention_step_i16_q15_checked(
        &workspace.embedding_output,
        attention_params,
        LinearAttentionStepWorkspace {
            q: &mut workspace.attention_q,
            k: &mut workspace.attention_k,
            v: &mut workspace.attention_v,
            context: &mut workspace.attention_context,
        },
        LinearAttentionState {
            state_kv: &mut workspace.state_kv,
            key_sums: &mut workspace.key_sums,
        },
        &mut workspace.attention_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_attention_step",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.embedding_output,
        &workspace.attention_output,
        &mut workspace.attention_residual,
    )?;

    let mlp_params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: &model.up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: &model.gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: &model.down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    gated_mlp_i16_q15_checked(
        &workspace.attention_residual,
        mlp_params,
        GatedMlpWorkspace {
            up: &mut workspace.mlp_up,
            gate: &mut workspace.mlp_gate,
            gated: &mut workspace.mlp_gated,
        },
        &mut workspace.mlp_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_mlp",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.attention_residual,
        &workspace.mlp_output,
        &mut workspace.block_output,
    )?;
    mini_transformer_output_row_for(&model.output_weights, &workspace.block_output)
}

fn mini_transformer_streaming_linear_ttt_nope_step(
    model: &MiniTransformerMlpModel,
    token: u8,
    workspace: &mut MiniTransformerStreamingLinearWorkspace,
    ttt_learning_rate_shift: u8,
) -> Result<(ByteVocabOutputRow, u64), TrainError> {
    mini_transformer_embedding_token_nope_q15(
        &model.embeddings,
        token,
        &mut workspace.embedding_output,
    )?;

    let attention_params = SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: &model.q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: &model.k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: &model.v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: &model.o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    };
    let delta_l1 = linear_attention_ttt_step_i16_q15_checked(
        &workspace.embedding_output,
        attention_params,
        LinearAttentionTttStepWorkspace {
            q: &mut workspace.attention_q,
            k: &mut workspace.attention_k,
            v: &mut workspace.attention_v,
            context: &mut workspace.attention_context,
            prediction: &mut workspace.attention_prediction,
        },
        LinearAttentionState {
            state_kv: &mut workspace.state_kv,
            key_sums: &mut workspace.key_sums,
        },
        &mut workspace.attention_output,
        ttt_learning_rate_shift,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_ttt_attention_step",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.embedding_output,
        &workspace.attention_output,
        &mut workspace.attention_residual,
    )?;

    let mlp_params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: &model.up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: &model.gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: &model.down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    gated_mlp_i16_q15_checked(
        &workspace.attention_residual,
        mlp_params,
        GatedMlpWorkspace {
            up: &mut workspace.mlp_up,
            gate: &mut workspace.mlp_gate,
            gated: &mut workspace.mlp_gated,
        },
        &mut workspace.mlp_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_ttt_mlp",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.attention_residual,
        &workspace.mlp_output,
        &mut workspace.block_output,
    )?;
    let row = mini_transformer_output_row_for(&model.output_weights, &workspace.block_output)?;
    Ok((row, delta_l1))
}

fn mini_transformer_embedding_token_nope_q15(
    embeddings: &[i16],
    token: u8,
    output: &mut [i16; MINI_TRANSFORMER_D_MODEL],
) -> Result<(), TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidConfig);
    }
    let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
    let row = embeddings
        .get(row_start..row_start + MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidModel("mini transformer embedding row"))?;
    output.copy_from_slice(row);
    Ok(())
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

fn residual_l1_i64(values: &[i64]) -> u64 {
    values.iter().fold(0_u64, |total, value| {
        total.saturating_add(value.unsigned_abs())
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LinearWeightGradientI64 {
    input_dim: usize,
    output_dim: usize,
    sample_count: usize,
    accumulators: Vec<i64>,
    residuals: Vec<i64>,
}

impl LinearWeightGradientI64 {
    fn new(input_dim: usize, output_dim: usize) -> Option<Self> {
        if input_dim == 0 || output_dim == 0 {
            return None;
        }
        let len = input_dim.checked_mul(output_dim)?;
        Some(Self {
            input_dim,
            output_dim,
            sample_count: 0,
            accumulators: vec![0_i64; len],
            residuals: vec![0_i64; len],
        })
    }

    fn clear(&mut self) {
        self.sample_count = 0;
        self.accumulators.fill(0);
    }

    fn as_train_core_workspace(&mut self) -> nsrl_train_core::LinearWeightGradientI64Workspace<'_> {
        nsrl_train_core::LinearWeightGradientI64Workspace {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            sample_count: self.sample_count,
            accumulators: &mut self.accumulators,
            residuals: &mut self.residuals,
        }
    }

    fn residual_l1(&self) -> u64 {
        residual_l1_i64(&self.residuals)
    }
}

fn accumulate_linear_weight_gradient_i64_prescaled(
    input: &[i16],
    scaled_grad_output: &[i32],
    gradient: &mut LinearWeightGradientI64,
) -> Result<(), TrainError> {
    let (result, sample_count) = {
        let mut workspace = gradient.as_train_core_workspace();
        let result = nsrl_train_core::accumulate_linear_weight_gradient_i64_prescaled(
            input,
            scaled_grad_output,
            &mut workspace,
        )
        .map_err(|error| {
            train_core_error_to_train_error(error, "linear_weight_gradient_accumulate")
        });
        (result, workspace.sample_count)
    };
    gradient.sample_count = sample_count;
    result
}

fn apply_linear_weight_gradient_i64_to_i8(
    gradient: &mut LinearWeightGradientI64,
    weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<LinearWeightUpdateStats, TrainError> {
    let (result, sample_count) = {
        let mut workspace = gradient.as_train_core_workspace();
        let result = nsrl_train_core::apply_linear_weight_gradient_i64_to_i8(
            &mut workspace,
            weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )
        .map_err(|error| train_core_error_to_train_error(error, "linear_weight_gradient_apply"));
        (result, workspace.sample_count)
    };
    gradient.sample_count = sample_count;
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GatedMlpWeightGradientI64 {
    down: LinearWeightGradientI64,
    up: LinearWeightGradientI64,
    gate: LinearWeightGradientI64,
}

impl GatedMlpWeightGradientI64 {
    fn new(d_model: usize, hidden_dim: usize) -> Option<Self> {
        Some(Self {
            down: LinearWeightGradientI64::new(hidden_dim, d_model)?,
            up: LinearWeightGradientI64::new(d_model, hidden_dim)?,
            gate: LinearWeightGradientI64::new(d_model, hidden_dim)?,
        })
    }

    fn clear(&mut self) {
        self.down.clear();
        self.up.clear();
        self.gate.clear();
    }

    fn residual_l1(&self) -> u64 {
        self.down
            .residual_l1()
            .saturating_add(self.up.residual_l1())
            .saturating_add(self.gate.residual_l1())
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_gated_mlp_weight_gradient_i64(
    input: &[i16],
    grad_output: &[i16],
    forward_gated: &[i16],
    grad_up: &[i16],
    grad_gate: &[i16],
    params: GatedMlpWeightUpdateParams<'_>,
    gradient: &mut GatedMlpWeightGradientI64,
    scaled_grad_output: &mut [i32],
) -> Result<(), TrainError> {
    if !params.is_valid() {
        return Err(TrainError::InvalidConfig);
    }
    let total = params
        .seq_len
        .checked_mul(params.d_model)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = params
        .seq_len
        .checked_mul(params.hidden_dim)
        .ok_or(TrainError::InvalidConfig)?;
    if input.len() != total
        || grad_output.len() != total
        || forward_gated.len() != hidden_total
        || grad_up.len() != hidden_total
        || grad_gate.len() != hidden_total
        || scaled_grad_output.len() < params.d_model.max(params.hidden_dim)
    {
        return Err(TrainError::InvalidConfig);
    }

    for token in 0..params.seq_len {
        let input_start = token
            .checked_mul(params.d_model)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_input_row"))?;
        let input_end = input_start
            .checked_add(params.d_model)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_input_end"))?;
        let hidden_start = token
            .checked_mul(params.hidden_dim)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_hidden_row"))?;
        let hidden_end = hidden_start
            .checked_add(params.hidden_dim)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_hidden_end"))?;

        let grad_row = &grad_output[input_start..input_end];
        if !grad_row.iter().any(|&gradient| gradient != 0) {
            continue;
        }

        linear_backward_prescale_grad_output_i16_i32_checked(
            grad_row,
            params.down_scales,
            &mut scaled_grad_output[..params.d_model],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_down_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &forward_gated[hidden_start..hidden_end],
            &scaled_grad_output[..params.d_model],
            &mut gradient.down,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_up[hidden_start..hidden_end],
            params.up_scales,
            &mut scaled_grad_output[..params.hidden_dim],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_up_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &input[input_start..input_end],
            &scaled_grad_output[..params.hidden_dim],
            &mut gradient.up,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_gate[hidden_start..hidden_end],
            params.gate_scales,
            &mut scaled_grad_output[..params.hidden_dim],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_gate_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &input[input_start..input_end],
            &scaled_grad_output[..params.hidden_dim],
            &mut gradient.gate,
        )?;
    }

    Ok(())
}

fn apply_gated_mlp_weight_gradient_i64_to_i8(
    gradient: &mut GatedMlpWeightGradientI64,
    up_weights: &mut [i8],
    gate_weights: &mut [i8],
    down_weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<GatedMlpWeightUpdateStats, TrainError> {
    Ok(GatedMlpWeightUpdateStats {
        down: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.down,
            down_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
        up: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.up,
            up_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
        gate: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.gate,
            gate_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerAttentionWeightGradientI64 {
    q: LinearWeightGradientI64,
    k: LinearWeightGradientI64,
    v: LinearWeightGradientI64,
    o: LinearWeightGradientI64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerRmsVectorGradientI64 {
    sample_count: usize,
    accumulators: Vec<i64>,
}

impl MiniTransformerRmsVectorGradientI64 {
    fn new() -> Self {
        Self {
            sample_count: 0,
            accumulators: vec![0_i64; MINI_TRANSFORMER_D_MODEL],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerRmsWeightGradientI64 {
    attention: MiniTransformerRmsVectorGradientI64,
    mlp: MiniTransformerRmsVectorGradientI64,
}

impl MiniTransformerRmsWeightGradientI64 {
    fn new() -> Self {
        Self {
            attention: MiniTransformerRmsVectorGradientI64::new(),
            mlp: MiniTransformerRmsVectorGradientI64::new(),
        }
    }
}

impl MiniTransformerAttentionWeightGradientI64 {
    fn new(d_model: usize) -> Option<Self> {
        Some(Self {
            q: LinearWeightGradientI64::new(d_model, d_model)?,
            k: LinearWeightGradientI64::new(d_model, d_model)?,
            v: LinearWeightGradientI64::new(d_model, d_model)?,
            o: LinearWeightGradientI64::new(d_model, d_model)?,
        })
    }

    fn clear(&mut self) {
        self.q.clear();
        self.k.clear();
        self.v.clear();
        self.o.clear();
    }

    fn residual_l1(&self) -> u64 {
        self.q
            .residual_l1()
            .saturating_add(self.k.residual_l1())
            .saturating_add(self.v.residual_l1())
            .saturating_add(self.o.residual_l1())
    }
}

fn accumulate_mini_transformer_attention_weight_gradient_i64(
    cache: &MiniTransformerBlockForwardCache,
    grad_attention_output: &[i16],
    grad_q: &[i16],
    grad_k: &[i16],
    grad_v: &[i16],
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    scaled_grad: &mut [i32],
) -> Result<(), TrainError> {
    let seq_len = cache.attention_norm.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || cache.attention_norm.len() != total
        || cache.attention_context.len() != total
        || grad_attention_output.len() != total
        || grad_q.len() != total
        || grad_k.len() != total
        || grad_v.len() != total
        || scaled_grad.len() < MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }

    for token in 0..seq_len {
        let row_start = token
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("attention_gradient_row"))?;
        let row_end = row_start
            .checked_add(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("attention_gradient_row_end"))?;
        let attention_input_row = &cache.attention_norm[row_start..row_end];

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_q[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_q_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.q,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_k[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_k_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.k,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_v[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_v_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.v,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_attention_output[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_o_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &cache.attention_context[row_start..row_end],
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.o,
        )?;
    }

    Ok(())
}

fn apply_mini_transformer_attention_weight_gradient_i64_to_i8(
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    model: &mut MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let final_layer_index = model
        .checked_transformer_layers()?
        .checked_sub(1)
        .ok_or(TrainError::InvalidConfig)?;
    apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
        gradient,
        model,
        final_layer_index,
        config,
    )
}

fn apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let transformer_layers = model.checked_transformer_layers()?;
    if layer_index >= transformer_layers {
        return Err(TrainError::InvalidConfig);
    }
    let attention_range = model.attention_weight_range(layer_index)?;
    let q = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.q,
        &mut model.q_weights[attention_range.clone()],
        config.learning_rate,
        config.attention_q_learning_rate_shift,
        true,
    )?;
    let k = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.k,
        &mut model.k_weights[attention_range.clone()],
        config.learning_rate,
        config.attention_qk_learning_rate_shift,
        true,
    )?;
    let use_vo_oracle = config.attention_vo_oracle && layer_index + 1 == transformer_layers;
    let (v, o) = if use_vo_oracle {
        gradient.v.clear();
        gradient.o.clear();
        (
            empty_linear_weight_update_stats(),
            empty_linear_weight_update_stats(),
        )
    } else {
        (
            apply_linear_weight_gradient_i64_to_i8(
                &mut gradient.v,
                &mut model.v_weights[attention_range.clone()],
                config.learning_rate,
                config.attention_learning_rate_shift,
                true,
            )?,
            apply_linear_weight_gradient_i64_to_i8(
                &mut gradient.o,
                &mut model.o_weights[attention_range],
                config.learning_rate,
                config.attention_learning_rate_shift,
                true,
            )?,
        )
    };

    let mut total = empty_linear_weight_update_stats();
    add_linear_weight_update_stats_checked(&mut total, q)?;
    add_linear_weight_update_stats_checked(&mut total, k)?;
    add_linear_weight_update_stats_checked(&mut total, v)?;
    add_linear_weight_update_stats_checked(&mut total, o)?;

    Ok(MiniTransformerAttentionWeightUpdateStats {
        q,
        k,
        v,
        o,
        gradient_saturation_count: total.gradient_saturation_count,
        zero_delta_count: total.zero_delta_count,
        weight_delta_l1: total.weight_delta_l1,
        grad_embedding_output: Vec::new(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MiniTransformerAttentionVoMatrix {
    Value,
    Output,
}

fn mini_transformer_attention_vo_oracle_update_i8_checked(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    step: i32,
) -> Result<(LinearWeightUpdateStats, LinearWeightUpdateStats), TrainError> {
    if starts.is_empty()
        || seq_len == 0
        || step <= 0
        || MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }

    let step = i64::from(step);
    let v = mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
        model,
        tokens,
        starts,
        seq_len,
        MiniTransformerAttentionVoMatrix::Value,
        step,
    )?;
    let o = mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
        model,
        tokens,
        starts,
        seq_len,
        MiniTransformerAttentionVoMatrix::Output,
        step,
    )?;
    Ok((v, o))
}

fn mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    matrix: MiniTransformerAttentionVoMatrix,
    step: i64,
) -> Result<LinearWeightUpdateStats, TrainError> {
    let len = mini_transformer_attention_weight_count()?;

    let mut stats = empty_linear_weight_update_stats();
    let mut current_loss =
        mini_transformer_total_probability_error_q15(tokens, starts, model, seq_len)?;
    for index in 0..len {
        let current = mini_transformer_attention_vo_weight(model, matrix, index)?;
        let mut best_value = current;
        let mut best_loss = current_loss;

        for direction in [1_i64, -1_i64] {
            let candidate_wide = i64::from(current)
                .checked_add(
                    direction
                        .checked_mul(step)
                        .ok_or(TrainError::CoreRejected("attention_vo_oracle_direction"))?,
                )
                .ok_or(TrainError::CoreRejected("attention_vo_oracle_candidate"))?;
            let candidate = saturate_i8(candidate_wide);
            if candidate == current {
                continue;
            }

            mini_transformer_set_attention_vo_weight(model, matrix, index, candidate)?;
            if let Ok(candidate_loss) =
                mini_transformer_total_probability_error_q15(tokens, starts, model, seq_len)
                && candidate_loss < best_loss
            {
                best_loss = candidate_loss;
                best_value = candidate;
            }
        }

        mini_transformer_set_attention_vo_weight(model, matrix, index, best_value)?;
        if best_value == current {
            stats.zero_delta_count =
                stats
                    .zero_delta_count
                    .checked_add(1)
                    .ok_or(TrainError::CoreRejected(
                        "attention_vo_oracle_zero_delta_count",
                    ))?;
        } else {
            let delta = i64::from(best_value) - i64::from(current);
            stats.weight_delta_l1 = stats
                .weight_delta_l1
                .checked_add(delta.unsigned_abs())
                .ok_or(TrainError::CoreRejected("attention_vo_oracle_delta_l1"))?;
            current_loss = best_loss;
        }
    }

    Ok(stats)
}

fn mini_transformer_attention_vo_weight(
    model: &MiniTransformerMlpModel,
    matrix: MiniTransformerAttentionVoMatrix,
    index: usize,
) -> Result<i8, TrainError> {
    let range = model.final_attention_weight_range()?;
    let absolute_index = range
        .start
        .checked_add(index)
        .ok_or(TrainError::InvalidConfig)?;
    if absolute_index >= range.end {
        return Err(TrainError::InvalidConfig);
    }
    match matrix {
        MiniTransformerAttentionVoMatrix::Value => model
            .v_weights
            .get(absolute_index)
            .copied()
            .ok_or(TrainError::InvalidConfig),
        MiniTransformerAttentionVoMatrix::Output => model
            .o_weights
            .get(absolute_index)
            .copied()
            .ok_or(TrainError::InvalidConfig),
    }
}

fn mini_transformer_set_attention_vo_weight(
    model: &mut MiniTransformerMlpModel,
    matrix: MiniTransformerAttentionVoMatrix,
    index: usize,
    value: i8,
) -> Result<(), TrainError> {
    let range = model.final_attention_weight_range()?;
    let absolute_index = range
        .start
        .checked_add(index)
        .ok_or(TrainError::InvalidConfig)?;
    if absolute_index >= range.end {
        return Err(TrainError::InvalidConfig);
    }
    let slot = match matrix {
        MiniTransformerAttentionVoMatrix::Value => model.v_weights.get_mut(absolute_index),
        MiniTransformerAttentionVoMatrix::Output => model.o_weights.get_mut(absolute_index),
    }
    .ok_or(TrainError::InvalidConfig)?;
    *slot = value;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerEmbeddingGradientI64 {
    sample_count: usize,
    token_accumulators: Vec<i64>,
    position_accumulators: Vec<i64>,
    token_residuals: Vec<i64>,
    position_residuals: Vec<i64>,
}

impl MiniTransformerEmbeddingGradientI64 {
    fn new(context_seq_len: usize) -> Option<Self> {
        let token_len = BYTE_VOCAB.checked_mul(MINI_TRANSFORMER_D_MODEL)?;
        let position_len = context_seq_len.checked_mul(MINI_TRANSFORMER_D_MODEL)?;
        Some(Self {
            sample_count: 0,
            token_accumulators: vec![0_i64; token_len],
            position_accumulators: vec![0_i64; position_len],
            token_residuals: vec![0_i64; token_len],
            position_residuals: vec![0_i64; position_len],
        })
    }

    fn clear(&mut self) {
        self.sample_count = 0;
        self.token_accumulators.fill(0);
        self.position_accumulators.fill(0);
    }

    fn residual_l1(&self, position_policy: MiniTransformerPositionPolicy) -> u64 {
        let token_l1 = residual_l1_i64(&self.token_residuals);
        if position_policy.uses_position_embeddings() {
            token_l1.saturating_add(residual_l1_i64(&self.position_residuals))
        } else {
            token_l1
        }
    }
}

fn accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    position_policy: MiniTransformerPositionPolicy,
    gradient: &mut MiniTransformerEmbeddingGradientI64,
) -> Result<(), TrainError> {
    if context.is_empty()
        || grad_embedding_output_q15.len()
            != context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
        || gradient.token_accumulators.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && gradient.position_accumulators.len()
            < context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    for (position, &token) in context.iter().enumerate() {
        let embedding_row_start = usize::from(token)
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("embedding_gradient_row"))?;
        let position_row_start = position
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("position_embedding_gradient_row"))?;
        let grad_row_start = position
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("embedding_gradient_grad_row"))?;
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let grad = i64::from(grad_embedding_output_q15[grad_row_start + dim]);
            if grad == 0 {
                continue;
            }
            let index = embedding_row_start
                .checked_add(dim)
                .ok_or(TrainError::CoreRejected("embedding_gradient_index"))?;
            gradient.token_accumulators[index] = gradient.token_accumulators[index]
                .checked_add(grad)
                .ok_or(TrainError::CoreRejected("embedding_gradient_accumulate"))?;
            if position_policy.uses_position_embeddings() {
                let position_index =
                    position_row_start
                        .checked_add(dim)
                        .ok_or(TrainError::CoreRejected(
                            "position_embedding_gradient_index",
                        ))?;
                gradient.position_accumulators[position_index] = gradient.position_accumulators
                    [position_index]
                    .checked_add(grad)
                    .ok_or(TrainError::CoreRejected(
                        "position_embedding_gradient_accumulate",
                    ))?;
            }
        }
    }

    gradient.sample_count = gradient
        .sample_count
        .checked_add(1)
        .ok_or(TrainError::CoreRejected("embedding_gradient_sample_count"))?;
    Ok(())
}

fn apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
    gradient: &mut MiniTransformerEmbeddingGradientI64,
    embeddings: &mut [i16],
    position_embeddings: &mut [i16],
    position_policy: MiniTransformerPositionPolicy,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != gradient.token_accumulators.len()
        || embeddings.len() != gradient.token_residuals.len()
        || learning_rate <= 0
        || embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && (position_embeddings.len() != gradient.position_accumulators.len()
            || position_embeddings.len() != gradient.position_residuals.len())
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = empty_softmax_update_stats();
    if gradient.sample_count == 0 {
        return Ok(stats);
    }

    apply_embedding_accumulators_i64_to_i16(
        &gradient.token_accumulators,
        &mut gradient.token_residuals,
        embeddings,
        gradient.sample_count,
        learning_rate,
        embedding_learning_rate_shift,
        &mut stats,
    )?;
    if position_policy.uses_position_embeddings() {
        apply_embedding_accumulators_i64_to_i16(
            &gradient.position_accumulators,
            &mut gradient.position_residuals,
            position_embeddings,
            gradient.sample_count,
            learning_rate,
            embedding_learning_rate_shift,
            &mut stats,
        )?;
    }

    gradient.clear();
    Ok(stats)
}

fn apply_embedding_accumulators_i64_to_i16(
    accumulators: &[i64],
    residuals: &mut [i64],
    embeddings: &mut [i16],
    sample_count: usize,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
    stats: &mut SoftmaxUpdateStats,
) -> Result<(), TrainError> {
    if accumulators.len() != residuals.len() || accumulators.len() != embeddings.len() {
        return Err(TrainError::InvalidConfig);
    }

    for ((raw_sum, residual), embedding) in accumulators
        .iter()
        .zip(residuals.iter_mut())
        .zip(embeddings.iter_mut())
    {
        if *raw_sum == 0 && *residual == 0 {
            continue;
        }
        let averaged = round_div_i64(*raw_sum, sample_count)?;
        let product = averaged
            .checked_mul(i64::from(learning_rate))
            .ok_or(TrainError::CoreRejected("embedding_gradient_apply_product"))?;
        let product = product
            .checked_add(*residual)
            .ok_or(TrainError::CoreRejected(
                "embedding_gradient_apply_residual",
            ))?;
        let scaled_update = round_shift_rhu_i64(product, embedding_learning_rate_shift);
        *residual =
            rounded_shift_residual_i64(product, scaled_update, embedding_learning_rate_shift)?;
        let delta = -scaled_update;
        if delta == 0 {
            stats.zero_delta_count = stats.zero_delta_count.saturating_add(1);
        }

        let previous = *embedding;
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainError::CoreRejected("embedding_gradient_apply_delta"))?;
        let clamped = saturate_i16(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats.gradient_saturation_count.saturating_add(1);
        }
        let applied_delta = i64::from(clamped) - i64::from(previous);
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .saturating_add(applied_delta.unsigned_abs());
        *embedding = clamped;
    }

    Ok(())
}

fn average_mini_transformer_batch_movement(
    base: &MiniTransformerMlpModel,
    model: &mut MiniTransformerMlpModel,
    divisor: usize,
    include_output_weights: bool,
    include_mlp_weights: bool,
    include_attention_weights: bool,
    include_embeddings: bool,
) -> Result<(), TrainError> {
    if divisor == 0 || base.context_seq_len != model.context_seq_len {
        return Err(TrainError::InvalidConfig);
    }

    if include_embeddings {
        average_i16_movement(&base.embeddings, &mut model.embeddings, divisor)?;
        average_i16_movement(
            &base.position_embeddings,
            &mut model.position_embeddings,
            divisor,
        )?;
    }
    if include_attention_weights {
        average_i8_movement(&base.q_weights, &mut model.q_weights, divisor)?;
        average_i8_movement(&base.k_weights, &mut model.k_weights, divisor)?;
        average_i8_movement(&base.v_weights, &mut model.v_weights, divisor)?;
        average_i8_movement(&base.o_weights, &mut model.o_weights, divisor)?;
    }
    if include_mlp_weights {
        average_i8_movement(&base.up_weights, &mut model.up_weights, divisor)?;
        average_i8_movement(&base.gate_weights, &mut model.gate_weights, divisor)?;
        average_i8_movement(&base.down_weights, &mut model.down_weights, divisor)?;
    }
    if include_output_weights {
        average_i8_movement(&base.output_weights, &mut model.output_weights, divisor)?;
    }
    Ok(())
}

fn average_i8_movement(base: &[i8], values: &mut [i8], divisor: usize) -> Result<(), TrainError> {
    if base.len() != values.len() || divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for (base, value) in base.iter().zip(values.iter_mut()) {
        let movement = i64::from(*value) - i64::from(*base);
        let averaged = round_div_i64(movement, divisor)?;
        *value = saturate_i8(i64::from(*base) + averaged);
    }

    Ok(())
}

fn average_i16_movement(
    base: &[i16],
    values: &mut [i16],
    divisor: usize,
) -> Result<(), TrainError> {
    if base.len() != values.len() || divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for (base, value) in base.iter().zip(values.iter_mut()) {
        let movement = i64::from(*value) - i64::from(*base);
        let averaged = round_div_i64(movement, divisor)?;
        *value = saturate_i16(i64::from(*base) + averaged);
    }

    Ok(())
}

fn round_div_i64(value: i64, divisor: usize) -> Result<i64, TrainError> {
    if divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let divisor = i64::try_from(divisor).map_err(|_| TrainError::InvalidConfig)?;
    let half = divisor / 2;
    if value >= 0 {
        Ok((value + half) / divisor)
    } else {
        Ok(-((-value + half) / divisor))
    }
}

fn rounded_shift_residual_i64(
    value: i64,
    shifted: i64,
    right_shift: u8,
) -> Result<i64, TrainError> {
    if right_shift == 0 {
        return Ok(0);
    }

    let applied = i128::from(shifted)
        .checked_shl(u32::from(right_shift))
        .ok_or(TrainError::CoreRejected("rounded_shift_residual_apply"))?;
    let residual = i128::from(value)
        .checked_sub(applied)
        .ok_or(TrainError::CoreRejected("rounded_shift_residual_subtract"))?;
    i64::try_from(residual).map_err(|_| TrainError::CoreRejected("rounded_shift_residual_range"))
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_validate_guard_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    epoch: usize,
    window_index: usize,
    epochs: usize,
) -> Result<(), TrainError> {
    if starts.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let len = starts.len();
    let candidates = [
        0,
        len / 4,
        len / 2,
        (len * 3) / 4,
        len - 1,
        window_index.saturating_sub(1),
        window_index,
        (window_index + 1).min(len - 1),
        if window_index + 1 < len {
            window_index + 1
        } else if epoch + 1 < epochs {
            0
        } else {
            window_index
        },
    ];

    let mut seen = [usize::MAX; 9];
    let mut seen_len = 0_usize;
    for &index in candidates.iter() {
        if index >= len || seen[..seen_len].contains(&index) {
            continue;
        }
        seen[seen_len] = index;
        seen_len += 1;

        let start = starts[index];
        mini_transformer_forward_for_attention_and_position(
            model,
            &tokens[start..start + seq_len],
            attention_kind,
            position_policy,
        )?;
    }

    Ok(())
}

fn mini_transformer_validate_batch_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<(), TrainError> {
    if starts.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for &start in starts {
        mini_transformer_forward_for_attention_and_position(
            model,
            &tokens[start..start + seq_len],
            attention_kind,
            position_policy,
        )?;
    }

    Ok(())
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

fn mini_transformer_block_backward_update_i8_checked(
    cache: &MiniTransformerBlockForwardCache,
    grad_block_output: &[i16],
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
) -> Result<MiniTransformerBlockBackwardUpdate, TrainError> {
    let seq_len = cache.block_output.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || grad_block_output.len() != total
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
        || cache.attention_output.len() != total
        || cache.attention_residual.len() != total
        || cache.mlp_norm.len() != total
        || cache.mlp_up.len() != hidden_total
        || cache.mlp_gate.len() != hidden_total
        || cache.mlp_gated.len() != hidden_total
        || cache.mlp_output.len() != total
        || cache.block_output.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    workspace.grad_mlp_output[..total].copy_from_slice(grad_block_output);

    let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
        &workspace.grad_mlp_output[..total],
        mini_transformer_mlp_params_for_layer(model, layer_index, seq_len)?,
        &cache.mlp_up,
        &cache.mlp_gate,
        GatedMlpBackwardScales {
            down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
            gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
        },
        GatedMlpBackwardWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_input_grad_gated,
            grad_up: &mut workspace.mlp_input_grad_up,
            grad_gate: &mut workspace.mlp_input_grad_gate,
            grad_up_input: &mut workspace.mlp_input_grad_up_input,
            grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
        },
        &mut workspace.grad_mlp_input,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_backward_input",
    ))?;

    let gradient_residual_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_mlp_output[..total],
        &workspace.grad_mlp_input[..total],
        &mut workspace.grad_attention_output[..total],
    )?;

    let up_or_gate_range = model.mlp_up_or_gate_weight_range(layer_index)?;
    let down_range = model.mlp_down_weight_range(layer_index)?;
    let mlp_update = gated_mlp_backward_weight_update_i8_checked(
        &cache.mlp_norm,
        &workspace.grad_mlp_output[..total],
        &cache.mlp_up,
        &cache.mlp_gate,
        &cache.mlp_gated,
        &mut model.up_weights[up_or_gate_range.clone()],
        &mut model.gate_weights[up_or_gate_range],
        &mut model.down_weights[down_range],
        GatedMlpWeightUpdateParams {
            up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            seq_len,
            d_model: MINI_TRANSFORMER_D_MODEL,
            hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.mlp_learning_rate_shift,
        },
        GatedMlpWeightUpdateWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_update_grad_gated,
            grad_up: &mut workspace.mlp_update_grad_up,
            grad_gate: &mut workspace.mlp_update_grad_gate,
        },
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_update",
    ))?;

    let attention_update = mini_transformer_attention_update_i8_checked(
        cache,
        model,
        layer_index,
        config,
        workspace,
        None,
    )?;

    let mut grad_input = vec![0_i16; total];
    let input_gradient_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_attention_output[..total],
        &workspace.grad_attention_norm_input[..total],
        &mut grad_input,
    )?;

    Ok(MiniTransformerBlockBackwardUpdate {
        mlp_update,
        attention_update,
        mlp_input_saturation_count,
        gradient_residual_saturation_count,
        input_gradient_saturation_count,
        grad_input,
    })
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_block_backward_accumulate_i64_checked(
    cache: &MiniTransformerBlockForwardCache,
    grad_block_output: &[i16],
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
    mlp_gradient: &mut GatedMlpWeightGradientI64,
    attention_gradient: &mut MiniTransformerAttentionWeightGradientI64,
    rms_gradient: &mut MiniTransformerRmsWeightGradientI64,
) -> Result<MiniTransformerBlockBackwardAccumulation, TrainError> {
    let seq_len = cache.block_output.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || grad_block_output.len() != total
        || cache.block_input.len() != total
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
        || cache.attention_output.len() != total
        || cache.attention_residual.len() != total
        || cache.mlp_norm.len() != total
        || cache.mlp_up.len() != hidden_total
        || cache.mlp_gate.len() != hidden_total
        || cache.mlp_gated.len() != hidden_total
        || cache.mlp_output.len() != total
        || cache.block_output.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let rms_weights = if model.rms_norm_enabled() {
        let range = model.rms_weight_range(layer_index)?;
        Some((
            model.attention_rms_weights[range.clone()].to_vec(),
            model.mlp_rms_weights[range].to_vec(),
        ))
    } else {
        None
    };
    workspace.grad_mlp_output[..total].copy_from_slice(grad_block_output);

    let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
        &workspace.grad_mlp_output[..total],
        mini_transformer_mlp_params_for_layer(model, layer_index, seq_len)?,
        &cache.mlp_up,
        &cache.mlp_gate,
        GatedMlpBackwardScales {
            down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
            gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
        },
        GatedMlpBackwardWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_input_grad_gated,
            grad_up: &mut workspace.mlp_input_grad_up,
            grad_gate: &mut workspace.mlp_input_grad_gate,
            grad_up_input: &mut workspace.mlp_input_grad_up_input,
            grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
        },
        &mut workspace.grad_mlp_input,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_backward_input",
    ))?;

    let mut grad_mlp_residual = vec![0_i16; total];
    let mlp_rms_backward_saturation_count = if let Some((_, mlp_weights)) = &rms_weights {
        mini_transformer_rms_norm_backward_rows(
            &cache.attention_residual,
            mlp_weights,
            &workspace.grad_mlp_input[..total],
            &mut grad_mlp_residual,
            &mut rms_gradient.mlp,
        )?
    } else {
        grad_mlp_residual.copy_from_slice(&workspace.grad_mlp_input[..total]);
        0
    };
    let gradient_residual_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_mlp_output[..total],
        &grad_mlp_residual,
        &mut workspace.grad_attention_output[..total],
    )?;

    accumulate_gated_mlp_weight_gradient_i64(
        &cache.mlp_norm,
        &workspace.grad_mlp_output[..total],
        &cache.mlp_gated,
        &workspace.mlp_input_grad_up,
        &workspace.mlp_input_grad_gate,
        GatedMlpWeightUpdateParams {
            up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            seq_len,
            d_model: MINI_TRANSFORMER_D_MODEL,
            hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.mlp_learning_rate_shift,
        },
        mlp_gradient,
        &mut workspace.mlp_scaled_grad,
    )?;

    let attention_update = mini_transformer_attention_update_i8_checked(
        cache,
        model,
        layer_index,
        config,
        workspace,
        Some(attention_gradient),
    )?;

    let mut grad_attention_input = vec![0_i16; total];
    let attention_rms_backward_saturation_count = if let Some((attention_weights, _)) = &rms_weights
    {
        mini_transformer_rms_norm_backward_rows(
            &cache.block_input,
            attention_weights,
            &workspace.grad_attention_norm_input[..total],
            &mut grad_attention_input,
            &mut rms_gradient.attention,
        )?
    } else {
        grad_attention_input.copy_from_slice(&workspace.grad_attention_norm_input[..total]);
        0
    };
    let mut grad_input = vec![0_i16; total];
    let input_gradient_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_attention_output[..total],
        &grad_attention_input,
        &mut grad_input,
    )?;

    Ok(MiniTransformerBlockBackwardAccumulation {
        mlp_input_saturation_count: mlp_input_saturation_count
            .saturating_add(mlp_rms_backward_saturation_count),
        attention_gradient_saturation_count: attention_update.gradient_saturation_count,
        gradient_residual_saturation_count,
        input_gradient_saturation_count: input_gradient_saturation_count
            .saturating_add(attention_rms_backward_saturation_count),
        grad_input,
    })
}

fn mini_transformer_attention_update_i8_checked(
    cache: &MiniTransformerBlockForwardCache,
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
    attention_gradient: Option<&mut MiniTransformerAttentionWeightGradientI64>,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let seq_len = cache.attention_norm.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let expected_probability_count = mini_transformer_attention_probability_count(seq_len)?;
    match config.attention_kind {
        MiniTransformerAttentionKind::Base2Softmax => {
            if cache.attention_probabilities_q15.len() != expected_probability_count {
                return Err(TrainError::InvalidConfig);
            }
        }
        MiniTransformerAttentionKind::Linear => {
            if !cache.attention_probabilities_q15.is_empty() {
                return Err(TrainError::InvalidConfig);
            }
        }
        MiniTransformerAttentionKind::LinearStreamingNope
        | MiniTransformerAttentionKind::LinearStreamingTttNope => {
            return Err(TrainError::InvalidConfig);
        }
    }
    let attention_range = model.attention_weight_range(layer_index)?;

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_output[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.o_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut workspace.grad_attention_context[row_start..row_end],
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_o_backward_input",
        ))?;
    }

    match config.attention_kind {
        MiniTransformerAttentionKind::Base2Softmax => {
            let grad_v = mini_transformer_attention_v_gradient_q15(
                seq_len,
                &cache.attention_probabilities_q15,
                &workspace.grad_attention_context,
            )?;
            let grad_probabilities = mini_transformer_attention_probability_gradient_q15(
                seq_len,
                &cache.attention_v,
                &workspace.grad_attention_context,
            )?;
            let grad_logits = mini_transformer_attention_logit_gradient_q15(
                seq_len,
                &cache.attention_probabilities_q15,
                &grad_probabilities,
            )?;
            let (grad_q, grad_k) = mini_transformer_attention_q_k_gradients_q15(
                seq_len,
                &cache.attention_q,
                &cache.attention_k,
                &grad_logits,
            )?;
            workspace.grad_attention_q[..total].copy_from_slice(&grad_q);
            workspace.grad_attention_k[..total].copy_from_slice(&grad_k);
            workspace.grad_attention_v[..total].copy_from_slice(&grad_v);
        }
        MiniTransformerAttentionKind::Linear => {
            mini_transformer_linear_attention_qkv_gradients_q15_workspace(
                seq_len,
                &cache.attention_q,
                &cache.attention_k,
                &cache.attention_v,
                workspace,
            )?;
        }
        MiniTransformerAttentionKind::LinearStreamingNope
        | MiniTransformerAttentionKind::LinearStreamingTttNope => {
            return Err(TrainError::InvalidConfig);
        }
    };
    let mut input_gradient_saturation_count = 0_usize;

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        let mut grad_q_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_k_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_v_input = [0_i16; MINI_TRANSFORMER_D_MODEL];

        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_q[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.q_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_q_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_q_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_k[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.k_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_k_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_k_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_v[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.v_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_v_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_v_backward_input",
        ))?;

        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let wide = i64::from(grad_q_input[dim])
                + i64::from(grad_k_input[dim])
                + i64::from(grad_v_input[dim]);
            let scaled = round_shift_rhu_i64(wide, MINI_TRANSFORMER_EMBEDDING_GRAD_FANIN_SHIFT);
            if scaled < i64::from(i16::MIN) || scaled > i64::from(i16::MAX) {
                input_gradient_saturation_count = input_gradient_saturation_count.saturating_add(1);
            }
            workspace.grad_attention_norm_input[row_start + dim] = saturate_i16(scaled);
        }
    }

    let mut total_stats = empty_linear_weight_update_stats();
    let mut q_total = empty_linear_weight_update_stats();
    let mut k_total = empty_linear_weight_update_stats();
    let mut v_total = empty_linear_weight_update_stats();
    let mut o_total = empty_linear_weight_update_stats();
    if let Some(attention_gradient) = attention_gradient {
        accumulate_mini_transformer_attention_weight_gradient_i64(
            cache,
            &workspace.grad_attention_output,
            &workspace.grad_attention_q,
            &workspace.grad_attention_k,
            &workspace.grad_attention_v,
            attention_gradient,
            &mut workspace.attention_scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )?;
    } else {
        for token in 0..seq_len {
            let row_start = token * MINI_TRANSFORMER_D_MODEL;
            let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
            let q_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_q[row_start..row_end],
                &mut model.q_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_q_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_q_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, q_stats)?;
            add_linear_weight_update_stats_checked(&mut q_total, q_stats)?;

            let k_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_k[row_start..row_end],
                &mut model.k_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_qk_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_k_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, k_stats)?;
            add_linear_weight_update_stats_checked(&mut k_total, k_stats)?;

            let v_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_v[row_start..row_end],
                &mut model.v_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_v_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, v_stats)?;
            add_linear_weight_update_stats_checked(&mut v_total, v_stats)?;

            let o_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_context[row_start..row_end],
                &workspace.grad_attention_output[row_start..row_end],
                &mut model.o_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_o_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, o_stats)?;
            add_linear_weight_update_stats_checked(&mut o_total, o_stats)?;
        }
    }

    Ok(MiniTransformerAttentionWeightUpdateStats {
        q: q_total,
        k: k_total,
        v: v_total,
        o: o_total,
        gradient_saturation_count: total_stats
            .gradient_saturation_count
            .saturating_add(input_gradient_saturation_count),
        zero_delta_count: total_stats.zero_delta_count,
        weight_delta_l1: total_stats.weight_delta_l1,
        grad_embedding_output: Vec::new(),
    })
}

#[cfg(test)]
type MiniTransformerLinearAttentionQkvGradients = (Vec<i16>, Vec<i16>, Vec<i16>);

fn mini_transformer_linear_attention_qkv_gradients_q15_workspace(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
) -> Result<(), TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0 || q.len() != total || k.len() != total || v.len() != total {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainError::InvalidConfig)?;
    let state_len = MINI_TRANSFORMER_HEADS
        .checked_mul(head_state_len)
        .ok_or(TrainError::InvalidConfig)?;
    workspace.linear_grad_q_acc[..total].fill(0);
    workspace.linear_grad_k_acc[..total].fill(0);
    workspace.linear_grad_v_acc[..total].fill(0);

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_head_start = head
            .checked_mul(seq_len)
            .and_then(|value| value.checked_mul(head_state_len))
            .ok_or(TrainError::InvalidConfig)?;
        let denom_head_start = head.checked_mul(seq_len).ok_or(TrainError::InvalidConfig)?;
        workspace.linear_grad_state_q15[..head_state_len].fill(0);
        let state_start = head
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let state_end = state_start
            .checked_add(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_start = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_end = key_sum_start
            .checked_add(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        workspace.attention_state_kv[state_start..state_end].fill(0);
        workspace.attention_key_sums[key_sum_start..key_sum_end].fill(0);

        for token in 0..seq_len {
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let state = &mut workspace.attention_state_kv[state_start..state_end];
            let key_sums = &mut workspace.attention_key_sums[key_sum_start..key_sum_end];

            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                key_sums[key_index] =
                    key_sums[key_index]
                        .checked_add(phi_key)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_key_sum",
                        ))?;
                let state_row_start = key_index
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_index, &value_value) in value.iter().enumerate() {
                    let product = phi_key.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_linear_attention_state_product"),
                    )?;
                    let state_index = state_row_start
                        .checked_add(value_index)
                        .ok_or(TrainError::InvalidConfig)?;
                    state[state_index] =
                        state[state_index]
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_state_accumulate",
                            ))?;
                }
            }

            let query = &q[row_start..row_end];
            let mut denominator = 0_i64;
            for (&query_value, &key_sum) in query.iter().zip(key_sums.iter()) {
                let product = mini_transformer_linear_attention_phi_i64(query_value)
                    .checked_mul(key_sum)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator_product",
                    ))?;
                denominator = denominator
                    .checked_add(product)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator",
                    ))?;
            }
            if denominator <= 0 {
                return Err(TrainError::CoreRejected(
                    "mini_transformer_linear_attention_nonpositive_denominator",
                ));
            }
            workspace.linear_denominators[denom_head_start + token] = denominator;
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainError::InvalidConfig)?,
                )
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            workspace.linear_prefix_states[snapshot_start..snapshot_end].copy_from_slice(state);
        }

        for query_index in 0..seq_len {
            let token = seq_len - 1 - query_index;
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let query = &q[row_start..row_end];
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let grad_row = &workspace.grad_attention_context[row_start..row_end];
            let denominator = workspace.linear_denominators[denom_head_start + token];
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainError::InvalidConfig)?,
                )
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let prefix_state = &workspace.linear_prefix_states[snapshot_start..snapshot_end];

            for key_dim in 0..head_dim {
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_q_numerator = 0_i64;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let product = i64::from(grad_value)
                        .checked_mul(prefix_state[state_index])
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_q_gradient_product",
                        ))?;
                    grad_q_numerator =
                        grad_q_numerator
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_q_gradient_accumulate",
                            ))?;
                }
                let target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                workspace.linear_grad_q_acc[target] = workspace.linear_grad_q_acc[target]
                    .checked_add(round_ratio_i64(grad_q_numerator, denominator)?)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_q_gradient_accumulate",
                    ))?;
            }

            for (key_dim, &query_value) in query.iter().enumerate() {
                let phi_query = mini_transformer_linear_attention_phi_i64(query_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let product = i64::from(grad_value)
                        .checked_mul(phi_query)
                        .and_then(|value| value.checked_mul(1_i64 << Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_product",
                        ))?;
                    let state_grad = round_ratio_i64(product, denominator)?;
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.linear_grad_state_q15[state_index] = workspace.linear_grad_state_q15
                        [state_index]
                        .checked_add(state_grad)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_accumulate",
                        ))?;
                }
            }

            for (key_dim, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_key_value = 0_i64;
                for (value_dim, &value_value) in value.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let state_grad = workspace.linear_grad_state_q15[state_index];
                    let grad_v_product =
                        state_grad
                            .checked_mul(phi_key)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_v_gradient_product",
                            ))?;
                    let v_target = row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.linear_grad_v_acc[v_target] = workspace.linear_grad_v_acc[v_target]
                        .checked_add(round_shift_rhu_i64(grad_v_product, Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_v_gradient_accumulate",
                        ))?;

                    let grad_k_product = state_grad.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_product",
                        ),
                    )?;
                    grad_key_value = grad_key_value
                        .checked_add(round_shift_rhu_i64(grad_k_product, Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_accumulate",
                        ))?;
                }
                let k_target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                workspace.linear_grad_k_acc[k_target] = workspace.linear_grad_k_acc[k_target]
                    .checked_add(grad_key_value)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_k_gradient_accumulate",
                    ))?;
            }
        }

        debug_assert!(state_len <= workspace.attention_state_kv.len());
    }

    for index in 0..total {
        workspace.grad_attention_q[index] = saturate_i16(workspace.linear_grad_q_acc[index]);
        workspace.grad_attention_k[index] = saturate_i16(workspace.linear_grad_k_acc[index]);
        workspace.grad_attention_v[index] = saturate_i16(workspace.linear_grad_v_acc[index]);
    }
    Ok(())
}

#[cfg(test)]
fn mini_transformer_linear_attention_qkv_gradients_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    grad_context: &[i16],
) -> Result<MiniTransformerLinearAttentionQkvGradients, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || q.len() != total
        || k.len() != total
        || v.len() != total
        || grad_context.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_q_acc = vec![0_i64; total];
    let mut grad_k_acc = vec![0_i64; total];
    let mut grad_v_acc = vec![0_i64; total];
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainError::InvalidConfig)?;

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let mut prefix_states = vec![
            0_i64;
            seq_len
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?
        ];
        let mut denominators = vec![0_i64; seq_len];
        let mut state = vec![0_i64; head_state_len];
        let mut key_sums = vec![0_i64; head_dim];

        for (token, denominator_slot) in denominators.iter_mut().enumerate().take(seq_len) {
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];

            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                key_sums[key_index] =
                    key_sums[key_index]
                        .checked_add(phi_key)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_key_sum",
                        ))?;
                let state_row_start = key_index
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_index, &value_value) in value.iter().enumerate() {
                    let product = phi_key.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_linear_attention_state_product"),
                    )?;
                    let state_index = state_row_start
                        .checked_add(value_index)
                        .ok_or(TrainError::InvalidConfig)?;
                    state[state_index] =
                        state[state_index]
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_state_accumulate",
                            ))?;
                }
            }

            let query = &q[row_start..row_end];
            let mut denominator = 0_i64;
            for (&query_value, &key_sum) in query.iter().zip(key_sums.iter()) {
                let product = mini_transformer_linear_attention_phi_i64(query_value)
                    .checked_mul(key_sum)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator_product",
                    ))?;
                denominator = denominator
                    .checked_add(product)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator",
                    ))?;
            }
            if denominator <= 0 {
                return Err(TrainError::CoreRejected(
                    "mini_transformer_linear_attention_nonpositive_denominator",
                ));
            }

            *denominator_slot = denominator;
            let snapshot_start = token
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            prefix_states[snapshot_start..snapshot_end].copy_from_slice(&state);
        }

        let mut grad_state_q15 = vec![0_i64; head_state_len];
        for query_index in 0..seq_len {
            let token = seq_len - 1 - query_index;
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let query = &q[row_start..row_end];
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let grad_row = &grad_context[row_start..row_end];
            let denominator = denominators[token];
            let snapshot_start = token
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let prefix_state = &prefix_states[snapshot_start..snapshot_end];

            for key_dim in 0..head_dim {
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_q_numerator = 0_i64;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let product = i64::from(grad_value)
                        .checked_mul(prefix_state[state_index])
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_q_gradient_product",
                        ))?;
                    grad_q_numerator =
                        grad_q_numerator
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_q_gradient_accumulate",
                            ))?;
                }
                let grad_q_value = round_ratio_i64(grad_q_numerator, denominator)?;
                let target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                grad_q_acc[target] = grad_q_acc[target].checked_add(grad_q_value).ok_or(
                    TrainError::CoreRejected(
                        "mini_transformer_linear_attention_q_gradient_accumulate",
                    ),
                )?;
            }

            for (key_dim, &query_value) in query.iter().enumerate() {
                let phi_query = mini_transformer_linear_attention_phi_i64(query_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let product = i64::from(grad_value)
                        .checked_mul(phi_query)
                        .and_then(|value| value.checked_mul(1_i64 << Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_product",
                        ))?;
                    let state_grad = round_ratio_i64(product, denominator)?;
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    grad_state_q15[state_index] = grad_state_q15[state_index]
                        .checked_add(state_grad)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_accumulate",
                        ))?;
                }
            }

            for (key_dim, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_key_value = 0_i64;
                for (value_dim, &value_value) in value.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let state_grad = grad_state_q15[state_index];
                    let grad_v_product =
                        state_grad
                            .checked_mul(phi_key)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_v_gradient_product",
                            ))?;
                    let grad_v_value = round_shift_rhu_i64(grad_v_product, Q15_SHIFT);
                    let v_target = row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    grad_v_acc[v_target] = grad_v_acc[v_target].checked_add(grad_v_value).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_v_gradient_accumulate",
                        ),
                    )?;

                    let grad_k_product = state_grad.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_product",
                        ),
                    )?;
                    let grad_k_value = round_shift_rhu_i64(grad_k_product, Q15_SHIFT);
                    grad_key_value = grad_key_value.checked_add(grad_k_value).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_accumulate",
                        ),
                    )?;
                }
                let k_target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                grad_k_acc[k_target] = grad_k_acc[k_target].checked_add(grad_key_value).ok_or(
                    TrainError::CoreRejected(
                        "mini_transformer_linear_attention_k_gradient_accumulate",
                    ),
                )?;
            }
        }
    }

    let mut grad_q = vec![0_i16; total];
    let mut grad_k = vec![0_i16; total];
    let mut grad_v = vec![0_i16; total];
    for index in 0..total {
        grad_q[index] = saturate_i16(grad_q_acc[index]);
        grad_k[index] = saturate_i16(grad_k_acc[index]);
        grad_v[index] = saturate_i16(grad_v_acc[index]);
    }

    Ok((grad_q, grad_k, grad_v))
}

fn mini_transformer_linear_attention_phi_i64(value: i16) -> i64 {
    i64::from(value) + 32769
}

fn round_ratio_i64(numerator: i64, denominator: i64) -> Result<i64, TrainError> {
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    if numerator >= 0 {
        numerator
            .checked_add(half)
            .map(|value| value / denominator)
            .ok_or(TrainError::CoreRejected("round_ratio_positive"))
    } else {
        numerator
            .checked_neg()
            .and_then(|value| value.checked_add(half))
            .map(|value| -(value / denominator))
            .ok_or(TrainError::CoreRejected("round_ratio_negative"))
    }
}

fn mini_transformer_attention_probability_gradient_q15(
    seq_len: usize,
    values: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0 || values.len() != total || grad_context.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_probabilities =
        vec![0_i16; mini_transformer_attention_probability_count(seq_len)?];
    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for query_index in 0..seq_len {
            let prob_row =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            for key_index in 0..seq_len {
                if key_index > query_index {
                    continue;
                }

                let query_start = query_index
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_add(head_offset))
                    .ok_or(TrainError::InvalidConfig)?;
                let key_start = key_index
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_add(head_offset))
                    .ok_or(TrainError::InvalidConfig)?;
                let mut acc = 0_i64;
                for dim in 0..head_dim {
                    let grad = grad_context[query_start + dim];
                    let value = values[key_start + dim];
                    let product = i64::from(grad).checked_mul(i64::from(value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_probability_gradient"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_probability_gradient_accumulate",
                    ))?;
                }

                grad_probabilities[prob_row + key_index] =
                    saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            }
        }
    }

    Ok(grad_probabilities)
}

fn mini_transformer_attention_logit_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_probabilities_q15: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0
        || probabilities_q15.len() != expected
        || grad_probabilities_q15.len() != expected
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_logits = vec![0_i16; expected];
    for head in 0..MINI_TRANSFORMER_HEADS {
        for query_index in 0..seq_len {
            let row_start =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            let row_end = row_start
                .checked_add(seq_len)
                .ok_or(TrainError::InvalidConfig)?;
            let probabilities = &probabilities_q15[row_start..row_end];
            let grad_probabilities = &grad_probabilities_q15[row_start..row_end];

            let mut weighted_grad = 0_i64;
            for key_index in 0..=query_index {
                let probability = probabilities[key_index];
                if probability < 0 {
                    return Err(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_negative_probability",
                    ));
                }
                let product = i64::from(grad_probabilities[key_index])
                    .checked_mul(i64::from(probability))
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_weighted_product",
                    ))?;
                weighted_grad =
                    weighted_grad
                        .checked_add(product)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_attention_logit_weighted_accumulate",
                        ))?;
            }
            let weighted_grad_q15 = round_shift_rhu_i64(weighted_grad, Q15_SHIFT);

            for key_index in 0..=query_index {
                let probability = probabilities[key_index];
                let centered = i64::from(grad_probabilities[key_index])
                    .checked_sub(weighted_grad_q15)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_center",
                    ))?;
                let product = i64::from(probability)
                    .checked_mul(centered)
                    .and_then(|value| value.checked_mul(i64::from(BASE2_SOFTMAX_LN2_Q15)))
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_gradient",
                    ))?;
                grad_logits[row_start + key_index] = saturate_i16(round_shift_rhu_i64(
                    product,
                    Q15_SHIFT.checked_mul(2).ok_or(TrainError::InvalidConfig)?,
                ));
            }
        }
    }

    Ok(grad_logits)
}

fn mini_transformer_attention_q_k_gradients_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    grad_logits_q15: &[i16],
) -> Result<(Vec<i16>, Vec<i16>), TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0 || q.len() != total || k.len() != total || grad_logits_q15.len() != expected {
        return Err(TrainError::InvalidConfig);
    }

    let sqrt_shift = sqrt_power_of_four_shift(head_dim).ok_or(TrainError::CoreRejected(
        "mini_transformer_attention_qk_sqrt_shift",
    ))?;
    let mut grad_q = vec![0_i16; total];
    let mut grad_k = vec![0_i16; total];

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for query_index in 0..seq_len {
            let prob_row =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            let query_start = query_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for key_index in 0..=query_index {
                    let grad_logit = grad_logits_q15[prob_row + key_index];
                    if grad_logit == 0 {
                        continue;
                    }
                    let key_start = key_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let key = k[key_start + dim];
                    let product = i64::from(grad_logit).checked_mul(i64::from(key)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_q_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_q_gradient_accumulate",
                    ))?;
                }
                grad_q[query_start + dim] = saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
            }
        }

        for key_index in 0..seq_len {
            let key_start = key_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for query_index in key_index..seq_len {
                    let prob_row = mini_transformer_attention_probability_row_start(
                        head,
                        query_index,
                        seq_len,
                    )?;
                    let grad_logit = grad_logits_q15[prob_row + key_index];
                    if grad_logit == 0 {
                        continue;
                    }
                    let query_start = query_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let query = q[query_start + dim];
                    let product = i64::from(grad_logit).checked_mul(i64::from(query)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_k_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_k_gradient_accumulate",
                    ))?;
                }
                grad_k[key_start + dim] = saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
            }
        }
    }

    Ok((grad_q, grad_k))
}

fn mini_transformer_attention_v_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0 || probabilities_q15.len() != expected || grad_context.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_v = vec![0_i16; total];
    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for key_index in 0..seq_len {
            let key_start = key_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for query_index in 0..seq_len {
                    let prob_row = mini_transformer_attention_probability_row_start(
                        head,
                        query_index,
                        seq_len,
                    )?;
                    let probability = probabilities_q15[prob_row + key_index];
                    if probability < 0 {
                        return Err(TrainError::CoreRejected(
                            "mini_transformer_attention_v_negative_probability",
                        ));
                    }
                    if probability == 0 {
                        continue;
                    }

                    let query_start = query_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let grad = grad_context[query_start + dim];
                    let product = i64::from(probability).checked_mul(i64::from(grad)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_v_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_v_gradient_accumulate",
                    ))?;
                }

                grad_v[key_start + dim] = saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            }
        }
    }

    Ok(grad_v)
}

fn empty_linear_weight_update_stats() -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    }
}

fn empty_softmax_update_stats() -> SoftmaxUpdateStats {
    SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    }
}

fn empty_gated_mlp_weight_update_stats() -> GatedMlpWeightUpdateStats {
    GatedMlpWeightUpdateStats {
        down: empty_linear_weight_update_stats(),
        up: empty_linear_weight_update_stats(),
        gate: empty_linear_weight_update_stats(),
    }
}

fn empty_mini_transformer_attention_weight_update_stats()
-> MiniTransformerAttentionWeightUpdateStats {
    MiniTransformerAttentionWeightUpdateStats {
        q: empty_linear_weight_update_stats(),
        k: empty_linear_weight_update_stats(),
        v: empty_linear_weight_update_stats(),
        o: empty_linear_weight_update_stats(),
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
        grad_embedding_output: Vec::new(),
    }
}

fn add_linear_weight_update_stats_checked(
    total: &mut LinearWeightUpdateStats,
    next: LinearWeightUpdateStats,
) -> Result<(), TrainError> {
    total.gradient_saturation_count = total
        .gradient_saturation_count
        .checked_add(next.gradient_saturation_count)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_saturation"))?;
    total.zero_delta_count = total
        .zero_delta_count
        .checked_add(next.zero_delta_count)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_zero_delta"))?;
    total.weight_delta_l1 = total
        .weight_delta_l1
        .checked_add(next.weight_delta_l1)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_delta_l1"))?;
    Ok(())
}

fn add_gated_mlp_weight_update_stats_checked(
    total: &mut GatedMlpWeightUpdateStats,
    next: GatedMlpWeightUpdateStats,
) -> Result<(), TrainError> {
    add_linear_weight_update_stats_checked(&mut total.down, next.down)?;
    add_linear_weight_update_stats_checked(&mut total.up, next.up)?;
    add_linear_weight_update_stats_checked(&mut total.gate, next.gate)?;
    Ok(())
}

fn add_mini_transformer_attention_weight_update_stats_checked(
    total: &mut MiniTransformerAttentionWeightUpdateStats,
    next: MiniTransformerAttentionWeightUpdateStats,
) -> Result<(), TrainError> {
    add_linear_weight_update_stats_checked(&mut total.q, next.q)?;
    add_linear_weight_update_stats_checked(&mut total.k, next.k)?;
    add_linear_weight_update_stats_checked(&mut total.v, next.v)?;
    add_linear_weight_update_stats_checked(&mut total.o, next.o)?;
    total.gradient_saturation_count = total
        .gradient_saturation_count
        .checked_add(next.gradient_saturation_count)
        .ok_or(TrainError::CoreRejected(
            "attention_weight_stats_saturation",
        ))?;
    total.zero_delta_count = total
        .zero_delta_count
        .checked_add(next.zero_delta_count)
        .ok_or(TrainError::CoreRejected(
            "attention_weight_stats_zero_delta",
        ))?;
    total.weight_delta_l1 = total
        .weight_delta_l1
        .checked_add(next.weight_delta_l1)
        .ok_or(TrainError::CoreRejected("attention_weight_stats_delta_l1"))?;
    Ok(())
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

pub fn mini_transformer_output_from_hidden_q15(
    model: &MiniTransformerMlpModel,
    hidden_q15: &[i16; MINI_TRANSFORMER_D_MODEL],
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let row = mini_transformer_output_row_for(&model.output_weights, hidden_q15)?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
    })
}

pub fn mini_transformer_output_gradient_to_hidden_q15(
    model: &MiniTransformerMlpModel,
    grad_output_q15: &[i16; BYTE_VOCAB],
) -> Result<[i16; MINI_TRANSFORMER_D_MODEL], TrainError> {
    let mut scaled_grad_output = [0_i32; BYTE_VOCAB];
    let mut grad_hidden_q15 = [0_i16; MINI_TRANSFORMER_D_MODEL];
    linear_backward_input_i16_i8_i16_per_channel_checked(
        grad_output_q15,
        LinearBackwardInputI16I8Params {
            weights: &model.output_weights,
            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
            grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: BYTE_VOCAB,
        },
        LinearBackwardInputWorkspace {
            scaled_grad_output: &mut scaled_grad_output,
        },
        &mut grad_hidden_q15,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_output_gradient_to_hidden",
    ))?;
    Ok(grad_hidden_q15)
}

#[derive(Debug, Clone)]
struct MiniTransformerBlockExpertLayerCache {
    base: MiniTransformerBlockForwardCache,
    latent_q15: Vec<i16>,
    adapted_output: Vec<i16>,
}

#[derive(Debug, Clone)]
struct MiniTransformerBlockExpertForwardCache {
    layers: Vec<MiniTransformerBlockExpertLayerCache>,
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
    hidden_saturation_count: usize,
}

fn block_expert_projection_sign(seed: u64, layer: usize, rank: usize, dim: usize) -> i64 {
    let mut value = seed
        ^ (layer as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (rank as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (dim as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    if value & 1 == 0 { 1 } else { -1 }
}

fn block_expert_layer_weight_range(
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
) -> Result<Range<usize>, TrainError> {
    if layer >= expert.transformer_layers {
        return Err(TrainError::InvalidConfig);
    }
    let per_layer = MINI_TRANSFORMER_D_MODEL
        .checked_mul(expert.rank)
        .ok_or(TrainError::InvalidConfig)?;
    let start = layer
        .checked_mul(per_layer)
        .ok_or(TrainError::InvalidConfig)?;
    Ok(start..start + per_layer)
}

fn block_expert_adapt_rows(
    base: &[i16],
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
) -> Result<(Vec<i16>, Vec<i16>, usize), TrainError> {
    if base.is_empty() || !base.len().is_multiple_of(MINI_TRANSFORMER_D_MODEL) {
        return Err(TrainError::InvalidConfig);
    }
    let rows = base.len() / MINI_TRANSFORMER_D_MODEL;
    let mut latent = vec![0_i16; rows * expert.rank];
    let mut output = vec![0_i16; base.len()];
    let weights = &expert.expansion_weights_q15[block_expert_layer_weight_range(expert, layer)?];
    let projection_shift = MINI_TRANSFORMER_D_MODEL.trailing_zeros() as u8;
    if 1_usize << u32::from(projection_shift) != MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidModel(
            "block expert d_model must be power of two",
        ));
    }
    let mut saturation_count = 0_usize;
    for row in 0..rows {
        let row_start = row * MINI_TRANSFORMER_D_MODEL;
        let latent_start = row * expert.rank;
        for rank in 0..expert.rank {
            let sum = (0..MINI_TRANSFORMER_D_MODEL)
                .map(|dim| {
                    i64::from(base[row_start + dim])
                        * block_expert_projection_sign(expert.projection_seed, layer, rank, dim)
                })
                .sum::<i64>();
            latent[latent_start + rank] = saturate_i16(round_shift_rhu_i64(sum, projection_shift));
        }
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let residual_acc = (0..expert.rank)
                .map(|rank| {
                    i64::from(latent[latent_start + rank])
                        * i64::from(weights[dim * expert.rank + rank])
                })
                .sum::<i64>();
            let raw = i64::from(base[row_start + dim]).saturating_add(round_shift_rhu_i64(
                residual_acc,
                Q15_SHIFT.saturating_add(expert.residual_shift),
            ));
            let adapted = saturate_i16(raw);
            saturation_count =
                saturation_count.saturating_add(usize::from(i64::from(adapted) != raw));
            output[row_start + dim] = adapted;
        }
    }
    Ok((output, latent, saturation_count))
}

fn mini_transformer_forward_with_block_expert(
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerBlockExpertForwardCache, TrainError> {
    expert.validate_for_model(model)?;
    if context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut layer_input = mini_transformer_embedding_sequence_with_position_policy_q15(
        &model.embeddings,
        &model.position_embeddings,
        context,
        position_policy,
    )?;
    let attention_weight_count = mini_transformer_attention_weight_count()?;
    let mlp_up_count = mini_transformer_mlp_up_or_gate_weight_count()?;
    let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
    let mut layer_caches = Vec::with_capacity(expert.transformer_layers);
    let mut hidden_saturation_count = 0_usize;
    for layer in 0..expert.transformer_layers {
        let attention_range = mini_transformer_layer_range(layer, attention_weight_count)?;
        let up_range = mini_transformer_layer_range(layer, mlp_up_count)?;
        let down_range = mini_transformer_layer_range(layer, mlp_down_count)?;
        let rms_range = if model.rms_norm_enabled() {
            Some(model.rms_weight_range(layer)?)
        } else {
            None
        };
        let base = mini_transformer_forward_block_for_attention_kind(
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
            &model.up_weights[up_range.clone()],
            &model.gate_weights[up_range],
            &model.down_weights[down_range],
            attention_kind,
        )?;
        let (adapted_output, latent_q15, saturations) =
            block_expert_adapt_rows(&base.block_output, expert, layer)?;
        hidden_saturation_count = hidden_saturation_count.saturating_add(saturations);
        layer_input = adapted_output.clone();
        layer_caches.push(MiniTransformerBlockExpertLayerCache {
            base,
            latent_q15,
            adapted_output,
        });
    }
    let last = layer_input
        .len()
        .checked_sub(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let mut output_features = [0_i16; MINI_TRANSFORMER_D_MODEL];
    output_features.copy_from_slice(&layer_input[last..last + MINI_TRANSFORMER_D_MODEL]);
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
    Ok(MiniTransformerBlockExpertForwardCache {
        layers: layer_caches,
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
        hidden_saturation_count,
    })
}

fn block_expert_backward_rows(
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
    cache: &MiniTransformerBlockExpertLayerCache,
    grad_adapted: &[i16],
    gradient_accumulators: &mut [i64],
) -> Result<Vec<i16>, TrainError> {
    if grad_adapted.len() != cache.base.block_output.len()
        || cache.adapted_output.len() != grad_adapted.len()
        || cache.latent_q15.len() * MINI_TRANSFORMER_D_MODEL != grad_adapted.len() * expert.rank
    {
        return Err(TrainError::InvalidConfig);
    }
    let range = block_expert_layer_weight_range(expert, layer)?;
    if gradient_accumulators.len() != expert.expansion_weights_q15.len() {
        return Err(TrainError::InvalidConfig);
    }
    let weights = &expert.expansion_weights_q15[range.clone()];
    let gradients = &mut gradient_accumulators[range];
    let rows = grad_adapted.len() / MINI_TRANSFORMER_D_MODEL;
    let projection_shift = MINI_TRANSFORMER_D_MODEL.trailing_zeros() as u8;
    let mut grad_base = vec![0_i16; grad_adapted.len()];
    for row in 0..rows {
        let row_start = row * MINI_TRANSFORMER_D_MODEL;
        let latent_start = row * expert.rank;
        let mut grad_latent = vec![0_i64; expert.rank];
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let index = row_start + dim;
            let grad = if cache.adapted_output[index] == i16::MIN
                || cache.adapted_output[index] == i16::MAX
            {
                0_i64
            } else {
                i64::from(grad_adapted[index])
            };
            for rank in 0..expert.rank {
                let weight_index = dim * expert.rank + rank;
                gradients[weight_index] = gradients[weight_index].saturating_add(
                    grad.saturating_mul(i64::from(cache.latent_q15[latent_start + rank])),
                );
                grad_latent[rank] = grad_latent[rank].saturating_add(round_shift_rhu_i64(
                    grad.saturating_mul(i64::from(weights[weight_index])),
                    Q15_SHIFT.saturating_add(expert.residual_shift),
                ));
            }
        }
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let projected = (0..expert.rank)
                .map(|rank| {
                    grad_latent[rank]
                        * block_expert_projection_sign(expert.projection_seed, layer, rank, dim)
                })
                .sum::<i64>();
            grad_base[row_start + dim] = saturate_i16(
                i64::from(grad_adapted[row_start + dim])
                    .saturating_add(round_shift_rhu_i64(projected, projection_shift)),
            );
        }
    }
    Ok(grad_base)
}

pub fn mini_transformer_next_token_row_with_block_expert(
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let cache = mini_transformer_forward_with_block_expert(
        model,
        expert,
        context,
        attention_kind,
        position_policy,
    )?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: cache.logits_q8,
        probabilities_q15: cache.probabilities_q15,
    })
}

pub fn evaluate_mini_transformer_block_expert(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpEvalConfig,
) -> Result<MiniTransformerBlockExpertMetrics, TrainError> {
    expert.validate_for_model(model)?;
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || config.attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: config.seq_len,
            stride: config.stride,
            max_windows: config.max_windows,
            attention_kind: config.attention_kind,
            position_policy: config.position_policy,
            ..MiniTransformerMlpTrainConfig::default()
        },
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut metrics = MiniTransformerBlockExpertMetrics {
        windows: starts.len(),
        mistakes: 0,
        probability_error_q15: 0,
        hidden_saturation_count: 0,
    };
    for start in starts {
        let end = start + config.seq_len;
        let cache = mini_transformer_forward_with_block_expert(
            model,
            expert,
            &tokens[start..end],
            config.attention_kind,
            config.position_policy,
        )?;
        metrics.mistakes = metrics.mistakes.saturating_add(usize::from(
            byte_argmax_i32(&cache.logits_q8) != tokens[end],
        ));
        metrics.probability_error_q15 =
            metrics
                .probability_error_q15
                .saturating_add(byte_sample_probability_error_q15(
                    &cache.probabilities_q15,
                    tokens[end],
                ));
        metrics.hidden_saturation_count = metrics
            .hidden_saturation_count
            .saturating_add(cache.hidden_saturation_count);
    }
    Ok(metrics)
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    train_mini_transformer_block_expert_with_layer_scope(
        tokens,
        model,
        expert,
        config,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert_with_layer_scope(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
    train_layer: Option<usize>,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
        tokens,
        model,
        expert,
        config,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        train_layer,
        false,
        MiniTransformerBlockExpertObjective::CrossEntropy,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
    train_layer: Option<usize>,
    bidirectional_loss_guard: bool,
    objective: MiniTransformerBlockExpertObjective,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    expert.validate_for_model(model)?;
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.seq_len != model.context_seq_len
        || batch_windows == 0
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind.uses_incremental_state()
        || train_layer.is_some_and(|layer| layer >= expert.transformer_layers)
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut update_residuals = vec![0_i64; expert.expansion_weights_q15.len()];
    let mut stats = MiniTransformerBlockExpertTrainStats {
        optimizer_steps: 0,
        accepted_forward_steps: 0,
        accepted_reverse_steps: 0,
        rejected_steps: 0,
        weight_delta_l1: 0,
        weight_saturation_count: 0,
        hidden_saturation_count: 0,
    };
    let last_start = (config.seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let mut workspace = MiniTransformerHostTrainCoreWorkspaceBuffers::new(config.seq_len)?;
    let mut frozen_model = model.clone();
    for _ in 0..config.epochs {
        for batch in starts.chunks(batch_windows) {
            let mut gradients = vec![0_i64; expert.expansion_weights_q15.len()];
            for &start in batch {
                let end = start + config.seq_len;
                let cache = mini_transformer_forward_with_block_expert(
                    model,
                    expert,
                    &tokens[start..end],
                    config.attention_kind,
                    config.position_policy,
                )?;
                stats.hidden_saturation_count = stats
                    .hidden_saturation_count
                    .saturating_add(cache.hidden_saturation_count);
                let mut grad_output =
                    byte_vocab_softmax_gradient_q15(&cache.probabilities_q15, tokens[end]);
                if objective == MiniTransformerBlockExpertObjective::ProbabilityError {
                    let target_probability =
                        i64::from(cache.probabilities_q15[usize::from(tokens[end])].max(0));
                    for gradient in &mut grad_output {
                        *gradient = (i64::from(*gradient).saturating_mul(target_probability)
                            / i64::from(i16::MAX))
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                            as i32;
                    }
                }
                apply_byte_argmax_margin_gradient_q15(
                    &mut grad_output,
                    &cache.logits_q8,
                    tokens[end],
                    config.argmax_margin_weight_q15,
                );
                let grad_output_q15 = byte_gradient_i32_to_i16(&grad_output);
                let grad_last =
                    mini_transformer_output_gradient_to_hidden_q15(model, &grad_output_q15)?;
                let mut grad_adapted = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                grad_adapted[last_start..last_start + MINI_TRANSFORMER_D_MODEL]
                    .copy_from_slice(&grad_last);
                let mut dummy =
                    MiniTransformerMapReduceBatchResult::new(config, expert.transformer_layers)?;
                for layer in (0..expert.transformer_layers).rev() {
                    let grad_base = block_expert_backward_rows(
                        expert,
                        layer,
                        &cache.layers[layer],
                        &grad_adapted,
                        &mut gradients,
                    )?;
                    let layer_config = mini_transformer_stacked_layer_runtime_config(
                        config,
                        layer,
                        expert.transformer_layers,
                    );
                    let backward = mini_transformer_block_backward_accumulate_i64_checked(
                        &cache.layers[layer].base,
                        &grad_base,
                        &mut frozen_model,
                        layer,
                        layer_config,
                        &mut workspace,
                        &mut dummy.mlp_weight_gradients[layer],
                        &mut dummy.attention_weight_gradients[layer],
                        &mut dummy.rms_weight_gradients[layer],
                    )?;
                    grad_adapted = backward.grad_input;
                }
            }
            let gradient_shift = learning_rate_shift
                .checked_add(Q15_SHIFT)
                .and_then(|shift| shift.checked_add(expert.residual_shift))
                .ok_or(TrainError::InvalidConfig)?;
            let denominator = i64::try_from(batch.len())
                .map_err(|_| TrainError::InvalidConfig)?
                .checked_shl(u32::from(gradient_shift))
                .ok_or(TrainError::InvalidConfig)?;
            let parameters_per_layer = MINI_TRANSFORMER_D_MODEL
                .checked_mul(expert.rank)
                .ok_or(TrainError::InvalidConfig)?;
            if bidirectional_loss_guard {
                let baseline_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                let original_weights = expert.expansion_weights_q15.clone();
                let original_residuals = update_residuals.clone();
                let mut forward_weights = original_weights.clone();
                let mut reverse_weights = original_weights.clone();
                let mut next_residuals = original_residuals.clone();
                let mut forward_saturations = vec![false; original_weights.len()];
                let mut reverse_saturations = vec![false; original_weights.len()];
                for index in 0..original_weights.len() {
                    if train_layer.is_some_and(|layer| index / parameters_per_layer != layer) {
                        continue;
                    }
                    let numerator = gradients[index]
                        .saturating_mul(learning_rate)
                        .saturating_add(original_residuals[index]);
                    let averaged = round_div_signed_i64(numerator, denominator)?;
                    next_residuals[index] =
                        numerator.saturating_sub(averaged.saturating_mul(denominator));
                    let update = averaged;
                    let previous = i64::from(original_weights[index]);
                    let forward_raw = previous.saturating_sub(update);
                    let reverse_raw = previous.saturating_add(update);
                    forward_weights[index] = saturate_i16(forward_raw);
                    reverse_weights[index] = saturate_i16(reverse_raw);
                    forward_saturations[index] = i64::from(forward_weights[index]) != forward_raw;
                    reverse_saturations[index] = i64::from(reverse_weights[index]) != reverse_raw;
                }
                expert.expansion_weights_q15 = forward_weights.clone();
                let forward_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                expert.expansion_weights_q15 = reverse_weights.clone();
                let reverse_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                let (selected, selected_saturations) = if forward_error < baseline_error
                    && forward_error <= reverse_error
                {
                    stats.accepted_forward_steps = stats.accepted_forward_steps.saturating_add(1);
                    (Some(forward_weights), Some(forward_saturations))
                } else if reverse_error < baseline_error && reverse_error < forward_error {
                    stats.accepted_reverse_steps = stats.accepted_reverse_steps.saturating_add(1);
                    (Some(reverse_weights), Some(reverse_saturations))
                } else {
                    stats.rejected_steps = stats.rejected_steps.saturating_add(1);
                    (None, None)
                };
                if let Some(selected) = selected {
                    for (index, (&previous, &next)) in
                        original_weights.iter().zip(selected.iter()).enumerate()
                    {
                        stats.weight_delta_l1 = stats
                            .weight_delta_l1
                            .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                        if selected_saturations
                            .as_ref()
                            .is_some_and(|values| values[index])
                        {
                            stats.weight_saturation_count =
                                stats.weight_saturation_count.saturating_add(1);
                            next_residuals[index] = 0;
                        }
                    }
                    expert.expansion_weights_q15 = selected;
                    update_residuals = next_residuals;
                } else {
                    expert.expansion_weights_q15 = original_weights;
                    update_residuals = original_residuals;
                }
            } else {
                for index in 0..expert.expansion_weights_q15.len() {
                    if train_layer.is_some_and(|layer| index / parameters_per_layer != layer) {
                        continue;
                    }
                    let numerator = gradients[index]
                        .saturating_mul(learning_rate)
                        .saturating_add(update_residuals[index]);
                    let averaged = round_div_signed_i64(numerator, denominator)?;
                    update_residuals[index] =
                        numerator.saturating_sub(averaged.saturating_mul(denominator));
                    let update = averaged;
                    let previous = expert.expansion_weights_q15[index];
                    let raw = i64::from(previous).saturating_sub(update);
                    let next = saturate_i16(raw);
                    if i64::from(next) != raw {
                        stats.weight_saturation_count =
                            stats.weight_saturation_count.saturating_add(1);
                        update_residuals[index] = 0;
                    }
                    stats.weight_delta_l1 = stats
                        .weight_delta_l1
                        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                    expert.expansion_weights_q15[index] = next;
                }
                stats.accepted_forward_steps = stats.accepted_forward_steps.saturating_add(1);
            }
            stats.optimizer_steps = stats.optimizer_steps.saturating_add(1);
        }
    }
    Ok(stats)
}

fn mini_transformer_block_expert_batch_error(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &start in starts {
        let end = start
            .checked_add(config.seq_len)
            .ok_or(TrainError::InvalidConfig)?;
        if end >= tokens.len() {
            return Err(TrainError::InvalidConfig);
        }
        let cache = mini_transformer_forward_with_block_expert(
            model,
            expert,
            &tokens[start..end],
            config.attention_kind,
            config.position_policy,
        )?;
        error = error.saturating_add(byte_sample_probability_error_q15(
            &cache.probabilities_q15,
            tokens[end],
        ));
    }
    Ok(error)
}

fn round_div_signed_i64(value: i64, denominator: i64) -> Result<i64, TrainError> {
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    Ok(if value >= 0 {
        value.saturating_add(half) / denominator
    } else {
        value.saturating_sub(half) / denominator
    })
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

#[cfg(test)]
fn select_byte_from_row(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
) -> Result<u8, TrainError> {
    Ok(select_byte_from_row_with_priors(
        logits_q8,
        probabilities_q15,
        decode,
        step_index,
        context,
        None,
    )?
    .token)
}

fn select_byte_from_row_with_priors(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<DecodeSelection, TrainError> {
    validate_decode_priors(decode, decode_priors)?;
    match decode.strategy {
        DecodeStrategy::Greedy => Ok(select_greedy_selection(
            logits_q8,
            probabilities_q15,
            decode,
            context,
            decode_priors,
        )),
        DecodeStrategy::Sample => sample_byte_from_probabilities_q15(
            logits_q8,
            probabilities_q15,
            decode,
            step_index,
            context,
            decode_priors,
        ),
    }
}

fn sample_byte_from_probabilities_q15(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<DecodeSelection, TrainError> {
    let candidate_set = decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let rejected_candidates = candidate_set.rejected_candidates;

    let mut mass = 0_u64;
    for &candidate in candidates.iter() {
        mass = mass.saturating_add(decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        ));
    }
    if mass == 0 {
        return Ok(decode_fallback_selection(
            logits_q8,
            probabilities_q15,
            decode,
            context,
            decode_priors,
        ));
    }

    let mut threshold = decode_sample_u64(decode.sample_seed, step_index, context) % mass;
    for &candidate in candidates.iter() {
        let weight = decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        );
        if threshold < weight {
            return Ok(DecodeSelection {
                token: candidate as u8,
                candidate_count: candidates.len(),
                rejected_candidates,
            });
        }
        threshold -= weight;
    }

    Ok(decode_fallback_selection(
        logits_q8,
        probabilities_q15,
        decode,
        context,
        decode_priors,
    ))
}

fn select_greedy_selection(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeSelection {
    if !decode_has_constraints(decode) {
        return DecodeSelection {
            token: byte_argmax_i32(logits_q8),
            candidate_count: BYTE_VOCAB,
            rejected_candidates: DecodeRejectStats::default(),
        };
    }

    decode_fallback_selection(logits_q8, probabilities_q15, decode, context, decode_priors)
}

fn decode_fallback_selection(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeSelection {
    let candidate_set = decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let token = if decode.repeat_window == 0
        && decode.repeat_penalty_shift == 0
        && !decode.corpus_prior
    {
        candidates
            .first()
            .copied()
            .unwrap_or_else(|| usize::from(byte_argmax_i32(logits_q8))) as u8
    } else {
        candidates
            .iter()
            .copied()
            .max_by_key(|&candidate| {
                (
                    decode_candidate_weight_q15(
                        probabilities_q15,
                        candidate,
                        decode,
                        context,
                        decode_priors,
                    ),
                    decode_effective_logit_q8(logits_q8, candidate, decode, context, decode_priors),
                    core::cmp::Reverse(candidate),
                )
            })
            .unwrap_or_else(|| usize::from(byte_argmax_i32(logits_q8))) as u8
    };
    DecodeSelection {
        token,
        candidate_count: candidates.len(),
        rejected_candidates: candidate_set.rejected_candidates,
    }
}

fn decode_candidates(
    logits_q8: &[i32; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeCandidateSet {
    let top_k = if decode.top_k == 0 || decode.top_k > BYTE_VOCAB {
        BYTE_VOCAB
    } else {
        decode.top_k
    };
    let mut rejected_candidates = DecodeRejectStats::default();
    let mut candidates = Vec::with_capacity(BYTE_VOCAB);
    for candidate in 0..BYTE_VOCAB {
        let token = candidate as u8;
        if decode.printable_only && !is_printable_decode_byte(token) {
            rejected_candidates.non_printable += 1;
            continue;
        }
        if decode.ascii_lower_only && !is_ascii_lower_text_decode_byte(token) {
            rejected_candidates.outside_ascii_lower += 1;
            continue;
        }
        if decode.max_repeat_run > 0
            && would_exceed_repeat_run(token, context, decode.max_repeat_run)
        {
            rejected_candidates.repeat_run += 1;
            continue;
        }
        if would_repeat_ngram(token, context, decode.no_repeat_ngram_order) {
            rejected_candidates.repeat_ngram += 1;
            continue;
        }
        if decode.strict_adjacency
            && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
            && !priors.allows_transition(previous, token)
        {
            rejected_candidates.adjacency += 1;
            continue;
        }
        candidates.push(candidate);
    }
    if candidates.len() > top_k {
        rejected_candidates.top_k_truncated += candidates.len() - top_k;
        candidates.select_nth_unstable_by(top_k, |&left, &right| {
            compare_byte_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
        });
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|&left, &right| {
        compare_byte_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
    });
    if candidates.is_empty() {
        candidates.push(usize::from(byte_argmax_i32(logits_q8)));
    }
    DecodeCandidateSet {
        candidates,
        rejected_candidates,
    }
}

fn compare_byte_decode_candidates(
    left: usize,
    right: usize,
    logits_q8: &[i32; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> core::cmp::Ordering {
    decode_effective_logit_q8(logits_q8, right, decode, context, decode_priors)
        .cmp(&decode_effective_logit_q8(
            logits_q8,
            left,
            decode,
            context,
            decode_priors,
        ))
        .then_with(|| left.cmp(&right))
}

fn decode_candidate_weight_q15(
    probabilities_q15: &[i16; BYTE_VOCAB],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> u64 {
    let mut weight = i32::from(probabilities_q15[candidate]).max(0) as u64;
    if decode.corpus_prior
        && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
    {
        let prior_q15 = priors.transition_probability_q15(previous, candidate as u8);
        let bonus = (weight.saturating_mul(u64::from(prior_q15))) >> Q15_SHIFT;
        weight = weight.saturating_add(bonus);
    }
    if decode.repeat_window > 0 && decode.repeat_penalty_shift > 0 {
        let repeat_count = recent_byte_count(candidate as u8, context, decode.repeat_window);
        let penalty_shift = repeat_count
            .saturating_mul(usize::from(decode.repeat_penalty_shift))
            .min(63);
        weight >>= penalty_shift;
    }
    weight
}

fn decode_has_constraints(decode: DecodeConfig) -> bool {
    decode.printable_only
        || decode.ascii_lower_only
        || decode.max_repeat_run > 0
        || decode.no_repeat_ngram_order > 1
        || decode.corpus_prior
        || decode.strict_adjacency
        || (decode.repeat_window > 0 && decode.repeat_penalty_shift > 0)
}

fn validate_decode_priors(
    decode: DecodeConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<(), TrainError> {
    if decode.corpus_prior
        && (decode.corpus_prior_order == 0
            || decode.corpus_prior_order > DEFAULT_CORPUS_PRIOR_ORDER)
    {
        return Err(TrainError::InvalidConfig);
    }
    if (decode.corpus_prior || decode.strict_adjacency) && decode_priors.is_none() {
        return Err(TrainError::InvalidConfig);
    }
    Ok(())
}

fn decode_effective_logit_q8(
    logits_q8: &[i32; BYTE_VOCAB],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> i32 {
    let mut logit = logits_q8[candidate];
    if decode.corpus_prior
        && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
    {
        let prior_q15 = i32::from(priors.transition_probability_q15(previous, candidate as u8));
        let shift = decode.corpus_prior_logit_shift.min(30);
        logit = logit.saturating_add(prior_q15 >> shift);
    }
    logit
}

fn is_printable_decode_byte(byte: u8) -> bool {
    byte == b'\n' || (b' '..=b'~').contains(&byte)
}

fn is_ascii_lower_text_decode_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'a'..=b'z'
            | b'0'..=b'9'
            | b'.'
            | b','
            | b';'
            | b':'
            | b'?'
            | b'!'
            | b'\''
            | b'-'
            | b' '
    )
}

fn recent_byte_count(candidate: u8, context: &[u8], repeat_window: usize) -> usize {
    context
        .iter()
        .rev()
        .take(repeat_window)
        .filter(|&&byte| byte == candidate)
        .count()
}

fn would_exceed_repeat_run(candidate: u8, context: &[u8], max_repeat_run: usize) -> bool {
    let run_len = context
        .iter()
        .rev()
        .take_while(|&&byte| byte == candidate)
        .count();
    run_len >= max_repeat_run
}

fn would_repeat_ngram<T: Copy + Eq>(candidate: T, context: &[T], ngram_order: usize) -> bool {
    if ngram_order < 2 || context.len() + 1 < ngram_order {
        return false;
    }

    let prefix_len = ngram_order - 1;
    let suffix_start = context.len() - prefix_len;
    let suffix = &context[suffix_start..];
    let search_end = context.len() + 1 - ngram_order;
    for start in 0..search_end {
        if &context[start..start + prefix_len] == suffix && context[start + prefix_len] == candidate
        {
            return true;
        }
    }
    false
}

fn decode_sample_u64(seed: u64, step_index: usize, context: &[u8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_bytes(&seed.to_le_bytes());
    hasher.update_usize(step_index);
    hasher.update_u8_slice(context);
    splitmix64(hasher.finish())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
mod tests {
    use super::*;
    #[test]
    fn checked_model_serialization_rejects_oversized_public_shapes() {
        if usize::BITS <= 32 {
            return;
        }

        let too_large = u32::MAX as usize + 1;

        let mini = MiniTransformerMlpModel {
            context_seq_len: too_large,
            embeddings: Vec::new(),
            position_embeddings: Vec::new(),
            attention_rms_weights: Vec::new(),
            mlp_rms_weights: Vec::new(),
            q_weights: Vec::new(),
            k_weights: Vec::new(),
            v_weights: Vec::new(),
            o_weights: Vec::new(),
            up_weights: Vec::new(),
            gate_weights: Vec::new(),
            down_weights: Vec::new(),
            output_weights: Vec::new(),
        };
        assert!(mini.try_to_bytes().is_err());
    }

    #[test]
    fn mini_transformer_adam_state_round_trips_separately_from_model() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
        let mut state =
            MiniTransformerAdamOptimizerState::new_for_model(&model, IntegerAdamConfig::default())
                .expect("new Adam state");
        state.step = 17;
        let last_parameter = state.parameter_count() - 1;
        state.first_moments[0] = -123;
        state.first_moments[last_parameter] = 456;
        state.second_moments[1] = 789;
        state.update_residuals[2] = -321;

        let bytes = state.to_bytes();
        let decoded =
            MiniTransformerAdamOptimizerState::from_bytes(&bytes).expect("decode Adam state");

        assert_eq!(decoded, state);
        decoded
            .validate_for_model(&model)
            .expect("state remains bound to model");
        assert_eq!(&bytes[..8], MINI_TRANSFORMER_ADAM_STATE_MAGIC);
        assert_ne!(&bytes[..8], MINI_TRANSFORMER_MODEL_MAGIC);
    }

    #[test]
    fn mini_transformer_adam_state_rejects_corruption_and_wrong_model() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let state =
            MiniTransformerAdamOptimizerState::new_for_model(&model, IntegerAdamConfig::default())
                .expect("new Adam state");
        let mut corrupt = state.to_bytes();
        corrupt[80] ^= 0x40;
        assert!(MiniTransformerAdamOptimizerState::from_bytes(&corrupt).is_err());
        assert!(MiniTransformerAdamOptimizerState::from_bytes(&corrupt[..64]).is_err());

        let mut changed_model = model.clone();
        changed_model.output_weights[0] = changed_model.output_weights[0].saturating_add(1);
        assert!(state.validate_for_model(&changed_model).is_err());

        let mut rebound = state.clone();
        rebound
            .bind_to_model(&changed_model)
            .expect("same-shape model can receive state after an accepted update");
        rebound
            .validate_for_model(&changed_model)
            .expect("rebound state/model pair");
    }

    #[test]
    fn mini_transformer_rms_model_round_trips_and_same_geometry_v4_stays_disabled() {
        let mut rms_model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
        rms_model.enable_rms_norm().expect("enable RMSNorm");
        assert!(rms_model.rms_norm_enabled());
        let rms_bytes = rms_model.to_bytes();
        assert_eq!(&rms_bytes[..8], MINI_TRANSFORMER_MODEL_MAGIC);
        let decoded = MiniTransformerMlpModel::from_bytes(&rms_bytes).expect("RMS model decode");
        assert_eq!(decoded, rms_model);

        let legacy_model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
        let mut legacy_bytes = legacy_model.to_bytes();
        legacy_bytes[..8].copy_from_slice(MINI_TRANSFORMER_LEGACY_MODEL_MAGIC);
        let decoded_legacy =
            MiniTransformerMlpModel::from_bytes(&legacy_bytes).expect("legacy v4 decode");
        assert_eq!(decoded_legacy, legacy_model);
        assert!(!decoded_legacy.rms_norm_enabled());
    }

    fn historical_v4_fixture_bytes(context_seq_len: usize) -> Vec<u8> {
        let embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
        let position_embeddings = vec![0_i16; context_seq_len * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
        let attention_count =
            MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_D_MODEL;
        let mut q_weights = vec![0_i8; attention_count];
        let mut k_weights = vec![0_i8; attention_count];
        let mut v_weights = vec![0_i8; attention_count];
        let mut o_weights = vec![0_i8; attention_count];
        for index in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
            let diagonal = index * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + index;
            q_weights[diagonal] = 1;
            k_weights[diagonal] = 1;
            v_weights[diagonal] = 1;
            o_weights[diagonal] = 1;
        }
        let up_weights =
            vec![0_i8; MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
        let gate_weights = up_weights.clone();
        let down_weights =
            vec![0_i8; MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM];
        let output_weights = vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];

        let mut embedding_hasher = StableHasher::new();
        embedding_hasher.update_i16_slice(&embeddings);
        embedding_hasher.update_i16_slice(&position_embeddings);
        let mut model_hasher = StableHasher::new();
        model_hasher.update_usize(context_seq_len);
        model_hasher.update_i16_slice(&embeddings);
        model_hasher.update_i16_slice(&position_embeddings);
        model_hasher.update_i8_slice(&q_weights);
        model_hasher.update_i8_slice(&k_weights);
        model_hasher.update_i8_slice(&v_weights);
        model_hasher.update_i8_slice(&o_weights);
        model_hasher.update_i8_slice(&up_weights);
        model_hasher.update_i8_slice(&gate_weights);
        model_hasher.update_i8_slice(&down_weights);
        model_hasher.update_i8_slice(&output_weights);

        let tensors = [
            q_weights.as_slice(),
            k_weights.as_slice(),
            v_weights.as_slice(),
            o_weights.as_slice(),
            up_weights.as_slice(),
            gate_weights.as_slice(),
            down_weights.as_slice(),
            output_weights.as_slice(),
        ];
        let counts = [
            embeddings.len(),
            position_embeddings.len(),
            q_weights.len(),
            k_weights.len(),
            v_weights.len(),
            o_weights.len(),
            up_weights.len(),
            gate_weights.len(),
            down_weights.len(),
            output_weights.len(),
        ];
        let hashes = [
            embedding_hasher.finish(),
            hash_i8_slice(&q_weights),
            hash_i8_slice(&k_weights),
            hash_i8_slice(&v_weights),
            hash_i8_slice(&o_weights),
            hash_three_i8_slices(&up_weights, &gate_weights, &down_weights),
            hash_i8_slice(&output_weights),
            model_hasher.finish(),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(MINI_TRANSFORMER_LEGACY_MODEL_MAGIC);
        for value in [
            BYTE_VOCAB,
            MINI_TRANSFORMER_LEGACY_V4_D_MODEL,
            MINI_TRANSFORMER_LEGACY_V4_HEADS,
            MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM,
            context_seq_len,
        ] {
            out.extend_from_slice(&(value as u32).to_le_bytes());
        }
        for count in counts {
            out.extend_from_slice(&(count as u64).to_le_bytes());
        }
        for hash in hashes {
            out.extend_from_slice(&hash.to_le_bytes());
        }
        for value in embeddings.iter().chain(position_embeddings.iter()) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for tensor in tensors {
            out.extend(tensor.iter().map(|&value| value as u8));
        }
        out
    }

    #[test]
    fn historical_v4_geometry_upgrades_for_eval_and_resume() {
        let bytes = historical_v4_fixture_bytes(4);
        let source_hash =
            MiniTransformerMlpModel::serialized_model_hash(&bytes).expect("serialized source hash");
        let upgraded = MiniTransformerMlpModel::from_bytes(&bytes).expect("historical V4 decode");
        assert_eq!(upgraded.context_seq_len, 4);
        assert_eq!(
            upgraded.embeddings.len(),
            BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        );
        assert_eq!(upgraded.transformer_layers(), 1);
        assert_ne!(upgraded.model_hash(), source_hash);
        assert_eq!(&upgraded.to_bytes()[..8], MINI_TRANSFORMER_MODEL_MAGIC);

        evaluate_mini_transformer_mlp_model(
            b"legacy checkpoint evaluation",
            &upgraded,
            MiniTransformerMlpEvalConfig {
                seq_len: 4,
                stride: 1,
                max_windows: Some(1),
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            },
        )
        .expect("upgraded model evaluates");

        run_mini_transformer_mlp_training_from_model(
            b"legacy checkpoint resume",
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                max_windows: Some(1),
                batch_windows: 1,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                ..MiniTransformerMlpTrainConfig::default()
            },
            upgraded,
        )
        .expect("upgraded model resumes training");
    }

    fn tiny_integer_adam_training_config() -> MiniTransformerMlpTrainConfig {
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            max_windows: Some(8),
            batch_windows: 4,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            batch_mode: MiniTransformerBatchMode::Serial,
            ..MiniTransformerMlpTrainConfig::default()
        }
    }

    #[test]
    fn mini_transformer_integer_adam_replay_is_deterministic() {
        let tokens = b"to be or not to be, that is the question";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };

        let left = run_mini_transformer_mlp_integer_adam_training(tokens, config, optimizer)
            .expect("left Adam run");
        let right = run_mini_transformer_mlp_integer_adam_training(tokens, config, optimizer)
            .expect("right Adam run");

        assert_eq!(left, right);
        assert_eq!(left.trace.updates, 8);
        assert_eq!(left.trace.optimizer_step, 2);
        assert_ne!(left.trace.initial_model_hash, left.trace.final_model_hash);
        assert!(left.trace.output_head_delta_l1 > 0);
        assert!(left.trace.mlp_delta_l1 > 0);
        assert!(left.trace.embedding_delta_l1 > 0);
        assert!(left.trace.attention_delta_l1 > 0);
        assert!(left.trace.attention_q_delta_l1 > 0);
        assert!(left.trace.attention_k_delta_l1 > 0);
        assert!(left.trace.attention_v_delta_l1 > 0);
        assert!(left.trace.attention_o_delta_l1 > 0);
        assert_eq!(left.trace.transformer_layers, 2);
        let json = left.trace.to_json_line();
        assert!(json.contains("\"architecture_profile\""));
        assert!(json.contains("\"argmax_margin_weight_q15\":0"));
        assert!(json.contains("\"attention_q\":"));
        assert!(json.contains("\"saturation\":"));
        left.optimizer_state
            .validate_for_model(&left.model)
            .expect("final optimizer binding");
    }

    #[cfg(feature = "mini-calibrated")]
    #[test]
    fn calibrated_suffix_memory_is_deterministic_and_uses_longest_suffix() {
        let tokens = b"abcaabdxabcaabdy";
        let mut first = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let mut replay = first.clone();

        let first_records =
            mini_transformer_install_ngram_cache(&mut first, tokens).expect("first cache");
        let replay_records =
            mini_transformer_install_ngram_cache(&mut replay, tokens).expect("replayed cache");

        assert_eq!(first_records, replay_records);
        assert_eq!(first.position_embeddings, replay.position_embeddings);
        assert_eq!(
            mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"zzabc"),
            Some(b'a')
        );
        assert_eq!(
            mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"zzabd"),
            Some(b'x')
        );
        assert_eq!(
            mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"qqqq"),
            Some(b'a')
        );
    }

    #[test]
    fn mini_transformer_integer_adam_updates_rmsnorm_gamma() {
        let tokens = b"Every thing that lives is holy, and each breath returns.";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        model.enable_rms_norm().expect("enable RMSNorm");
        let initial_attention_gamma = model.attention_rms_weights.clone();
        let initial_mlp_gamma = model.mlp_rms_weights.clone();

        let run = run_mini_transformer_mlp_integer_adam_training_from_model(
            tokens, config, optimizer, model, None,
        )
        .expect("RMSNorm Adam run");

        assert!(run.model.rms_norm_enabled());
        assert_ne!(run.model.attention_rms_weights, initial_attention_gamma);
        assert_ne!(run.model.mlp_rms_weights, initial_mlp_gamma);
        assert!(run.trace.rms_norm_delta_l1 > 0);
        run.optimizer_state
            .validate_for_model(&run.model)
            .expect("RMS optimizer binding");
    }

    #[test]
    fn mini_transformer_rmsnorm_scope_updates_only_internal_gamma() {
        let tokens = b"The imagination is not a state; it is existence itself.";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        model.enable_rms_norm().expect("enable RMSNorm");
        let initial = model.clone();

        let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
            tokens,
            config,
            optimizer,
            model,
            None,
            MiniTransformerAdamTrainScope::RmsNorm,
        )
        .expect("RMSNorm-only Adam run");

        assert_eq!(
            run.trace.train_scope,
            MiniTransformerAdamTrainScope::RmsNorm
        );
        assert!(run.trace.rms_norm_delta_l1 > 0);
        assert_eq!(run.trace.output_head_delta_l1, 0);
        assert_eq!(run.trace.mlp_delta_l1, 0);
        assert_eq!(run.trace.embedding_delta_l1, 0);
        assert_eq!(run.trace.attention_delta_l1, 0);
        assert_ne!(
            run.model.attention_rms_weights,
            initial.attention_rms_weights
        );
        assert_ne!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
        assert_eq!(run.model.embeddings, initial.embeddings);
        assert_eq!(run.model.position_embeddings, initial.position_embeddings);
        assert_eq!(run.model.q_weights, initial.q_weights);
        assert_eq!(run.model.k_weights, initial.k_weights);
        assert_eq!(run.model.v_weights, initial.v_weights);
        assert_eq!(run.model.o_weights, initial.o_weights);
        assert_eq!(run.model.up_weights, initial.up_weights);
        assert_eq!(run.model.gate_weights, initial.gate_weights);
        assert_eq!(run.model.down_weights, initial.down_weights);
        assert_eq!(run.model.output_weights, initial.output_weights);
    }

    #[test]
    fn mini_transformer_final_mlp_scope_freezes_shared_trunk() {
        let tokens = b"The tygers of wrath are wiser than the horses of instruction.";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        model.enable_rms_norm().expect("enable RMSNorm");
        let initial = model.clone();
        let layers = model.transformer_layers();
        assert!(layers > 1);
        let final_up = model
            .mlp_up_or_gate_weight_range(layers - 1)
            .expect("final up range");
        let final_down = model
            .mlp_down_weight_range(layers - 1)
            .expect("final down range");

        let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
            tokens,
            config,
            optimizer,
            model,
            None,
            MiniTransformerAdamTrainScope::FinalMlp,
        )
        .expect("final MLP Adam run");

        assert_eq!(
            run.trace.train_scope,
            MiniTransformerAdamTrainScope::FinalMlp
        );
        assert_eq!(run.trace.output_head_delta_l1, 0);
        assert_eq!(run.trace.embedding_delta_l1, 0);
        assert_eq!(run.trace.rms_norm_delta_l1, 0);
        assert_eq!(run.trace.attention_delta_l1, 0);
        assert!(run.trace.mlp_delta_l1 > 0);
        assert_eq!(run.model.embeddings, initial.embeddings);
        assert_eq!(run.model.position_embeddings, initial.position_embeddings);
        assert_eq!(
            run.model.attention_rms_weights,
            initial.attention_rms_weights
        );
        assert_eq!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
        assert_eq!(run.model.q_weights, initial.q_weights);
        assert_eq!(run.model.k_weights, initial.k_weights);
        assert_eq!(run.model.v_weights, initial.v_weights);
        assert_eq!(run.model.o_weights, initial.o_weights);
        assert_eq!(run.model.output_weights, initial.output_weights);
        assert_eq!(
            run.model.up_weights[..final_up.start],
            initial.up_weights[..final_up.start]
        );
        assert_eq!(
            run.model.gate_weights[..final_up.start],
            initial.gate_weights[..final_up.start]
        );
        assert_eq!(
            run.model.down_weights[..final_down.start],
            initial.down_weights[..final_down.start]
        );
        assert_ne!(
            run.model.up_weights[final_up.clone()],
            initial.up_weights[final_up]
        );
        assert_ne!(
            run.model.down_weights[final_down.clone()],
            initial.down_weights[final_down]
        );
        run.optimizer_state
            .validate_for_model(&run.model)
            .expect("final MLP optimizer binding");
    }

    #[test]
    fn mini_transformer_final_mlp_output_scope_updates_expert_only() {
        let tokens = b"Energy is eternal delight, carried by the speaking flame.";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        model.enable_rms_norm().expect("enable RMSNorm");
        let initial = model.clone();

        let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
            tokens,
            config,
            optimizer,
            model,
            None,
            MiniTransformerAdamTrainScope::FinalMlpAndOutput,
        )
        .expect("final MLP and output Adam run");

        assert_eq!(
            run.trace.train_scope,
            MiniTransformerAdamTrainScope::FinalMlpAndOutput
        );
        assert!(run.trace.output_head_delta_l1 > 0);
        assert!(run.trace.mlp_delta_l1 > 0);
        assert_eq!(run.trace.embedding_delta_l1, 0);
        assert_eq!(run.trace.rms_norm_delta_l1, 0);
        assert_eq!(run.trace.attention_delta_l1, 0);
        assert_eq!(run.model.embeddings, initial.embeddings);
        assert_eq!(run.model.position_embeddings, initial.position_embeddings);
        assert_eq!(
            run.model.attention_rms_weights,
            initial.attention_rms_weights
        );
        assert_eq!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
        assert_eq!(run.model.q_weights, initial.q_weights);
        assert_eq!(run.model.k_weights, initial.k_weights);
        assert_eq!(run.model.v_weights, initial.v_weights);
        assert_eq!(run.model.o_weights, initial.o_weights);
        assert_ne!(run.model.output_weights, initial.output_weights);
    }

    #[test]
    fn mini_transformer_output_scope_freezes_all_hidden_layers() {
        let tokens = b"All deities reside in the human breast.";
        let config = tiny_integer_adam_training_config();
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        let initial = model.clone();
        let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
            tokens,
            config,
            optimizer,
            model,
            None,
            MiniTransformerAdamTrainScope::Output,
        )
        .expect("output-only Adam run");

        assert_eq!(run.trace.train_scope, MiniTransformerAdamTrainScope::Output);
        assert!(run.trace.output_head_delta_l1 > 0);
        assert_eq!(run.trace.mlp_delta_l1, 0);
        assert_eq!(run.trace.embedding_delta_l1, 0);
        assert_eq!(run.trace.rms_norm_delta_l1, 0);
        assert_eq!(run.trace.attention_delta_l1, 0);
        assert_eq!(run.model.embeddings, initial.embeddings);
        assert_eq!(run.model.q_weights, initial.q_weights);
        assert_eq!(run.model.up_weights, initial.up_weights);
        assert_ne!(run.model.output_weights, initial.output_weights);
    }

    #[test]
    fn mini_transformer_rmsnorm_adam_serial_map_reduce_parity() {
        let tokens = b"The road of excess leads to the palace of wisdom.";
        let serial_config = MiniTransformerMlpTrainConfig {
            max_windows: Some(4),
            ..tiny_integer_adam_training_config()
        };
        let map_reduce_config = MiniTransformerMlpTrainConfig {
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 2,
            ..serial_config
        };
        let optimizer = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(serial_config.seq_len);
        model.enable_rms_norm().expect("enable RMSNorm");
        let serial = run_mini_transformer_mlp_integer_adam_training_from_model(
            tokens,
            serial_config,
            optimizer,
            model.clone(),
            None,
        )
        .expect("serial RMSNorm run");
        let map_reduce = run_mini_transformer_mlp_integer_adam_training_from_model(
            tokens,
            map_reduce_config,
            optimizer,
            model,
            None,
        )
        .expect("map-reduce RMSNorm run");

        assert_eq!(serial.model, map_reduce.model);
        assert_eq!(serial.optimizer_state, map_reduce.optimizer_state);
        assert_eq!(
            serial.trace.rms_norm_delta_l1,
            map_reduce.trace.rms_norm_delta_l1
        );
    }

    #[test]
    fn mini_transformer_integer_adam_serial_map_reduce_parity() {
        let tokens = b"Tyger Tyger, burning bright, in the forests of the night";
        let serial_config = tiny_integer_adam_training_config();
        let map_reduce_config = MiniTransformerMlpTrainConfig {
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 2,
            ..serial_config
        };
        let optimizer = IntegerAdamConfig {
            step_shift: 1,
            ..IntegerAdamConfig::default()
        };

        let serial =
            run_mini_transformer_mlp_integer_adam_training(tokens, serial_config, optimizer)
                .expect("serial Adam run");
        let map_reduce =
            run_mini_transformer_mlp_integer_adam_training(tokens, map_reduce_config, optimizer)
                .expect("map-reduce Adam run");

        assert_eq!(serial.model, map_reduce.model);
        assert_eq!(serial.optimizer_state, map_reduce.optimizer_state);
        assert_eq!(
            serial.trace.final_model_hash,
            map_reduce.trace.final_model_hash
        );
        assert_eq!(
            serial.trace.optimizer_state_hash,
            map_reduce.trace.optimizer_state_hash
        );
    }

    #[test]
    fn mini_transformer_integer_adam_resume_matches_uninterrupted_training() {
        let tokens = b"Shall I compare thee to a summer's day? Thou art more lovely.";
        let one_epoch = tiny_integer_adam_training_config();
        let two_epochs = MiniTransformerMlpTrainConfig {
            epochs: 2,
            ..one_epoch
        };
        let optimizer = IntegerAdamConfig {
            step_shift: 1,
            ..IntegerAdamConfig::default()
        };
        let uninterrupted =
            run_mini_transformer_mlp_integer_adam_training(tokens, two_epochs, optimizer)
                .expect("uninterrupted Adam run");
        let first = run_mini_transformer_mlp_integer_adam_training(tokens, one_epoch, optimizer)
            .expect("first resumed epoch");
        let state_bytes = first.optimizer_state.to_bytes();
        let resumed_state = MiniTransformerAdamOptimizerState::from_bytes(&state_bytes)
            .expect("resume state decode");
        let resumed = run_mini_transformer_mlp_integer_adam_training_from_model(
            tokens,
            one_epoch,
            optimizer,
            MiniTransformerMlpModel::from_bytes(&first.model.to_bytes())
                .expect("resume model decode"),
            Some(resumed_state),
        )
        .expect("resumed Adam run");

        assert_eq!(uninterrupted.model, resumed.model);
        assert_eq!(uninterrupted.optimizer_state, resumed.optimizer_state);
        assert_eq!(
            uninterrupted.trace.final_model_hash,
            resumed.trace.final_model_hash
        );
        assert_eq!(
            uninterrupted.trace.optimizer_step,
            resumed.trace.optimizer_step
        );
    }

    #[test]
    fn byte_target_frequency_weights_only_downweight_common_targets() {
        let tokens = [b'x', b'a', b'y', b'a', b'z', b'a', b'w', b'b'];
        let weights = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 2, 4096)
            .expect("byte target frequency weights");

        assert!(weights[usize::from(b'a')] < i16::MAX);
        assert!(weights[usize::from(b'a')] >= 4096);
        assert_eq!(weights[usize::from(b'b')], i16::MAX);
        assert_eq!(weights[usize::from(b'c')], i16::MAX);

        let disabled = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 0, 4096)
            .expect("disabled byte target frequency weights");
        assert!(disabled.iter().all(|&weight| weight == i16::MAX));
    }

    #[test]
    fn byte_argmax_margin_gradient_pushes_target_against_best_competitor() {
        let mut gradient = [0_i32; BYTE_VOCAB];
        let mut logits = [0_i32; BYTE_VOCAB];
        logits[usize::from(b'a')] = 10;
        logits[usize::from(b'b')] = 12;
        logits[usize::from(b'c')] = 12;

        apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);

        assert!(gradient[usize::from(b'a')] < 0);
        assert!(gradient[usize::from(b'b')] > 0);
        assert_eq!(gradient[usize::from(b'c')], 0);

        let pushed_target = gradient[usize::from(b'a')];
        let pushed_competitor = gradient[usize::from(b'b')];
        logits[usize::from(b'a')] = 13;
        apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);
        assert_eq!(gradient[usize::from(b'a')], pushed_target);
        assert_eq!(gradient[usize::from(b'b')], pushed_competitor);
    }

    #[test]
    fn sample_decode_is_deterministic_and_can_escape_argmax() {
        let logits = [0_i32; BYTE_VOCAB];
        let mut probabilities = [0_i16; BYTE_VOCAB];
        for probability in probabilities.iter_mut().take(4) {
            *probability = 8192;
        }
        let decode = DecodeConfig {
            strategy: DecodeStrategy::Sample,
            sample_seed: 7,
            top_k: 4,
            ..DecodeConfig::greedy()
        };

        let left = select_byte_from_row(&logits, &probabilities, decode, 3, b"context")
            .expect("sample left");
        let right = select_byte_from_row(&logits, &probabilities, decode, 3, b"context")
            .expect("sample right");

        assert_eq!(left, right);
        assert!(left < 4);
        assert!((0..64).any(|seed| {
            let decode = DecodeConfig {
                strategy: DecodeStrategy::Sample,
                sample_seed: seed,
                top_k: 4,
                ..DecodeConfig::greedy()
            };
            select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
                .is_ok_and(|token| token != 0 && token < 4)
        }));
    }

    #[test]
    fn printable_decode_filters_control_bytes() {
        let mut logits = [0_i32; BYTE_VOCAB];
        let mut probabilities = [0_i16; BYTE_VOCAB];
        logits[0] = 1000;
        probabilities[0] = 20_000;
        logits[usize::from(b'A')] = 900;
        probabilities[usize::from(b'A')] = 10_000;
        let decode = DecodeConfig {
            printable_only: true,
            ..DecodeConfig::greedy()
        };

        let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
            .expect("printable decode");

        assert_eq!(token, b'A');
    }

    #[test]
    fn ascii_lower_decode_filters_outside_curriculum_bytes() {
        let mut logits = [0_i32; BYTE_VOCAB];
        let probabilities = [1_i16; BYTE_VOCAB];
        logits[usize::from(b'Z')] = 1000;
        logits[usize::from(b'@')] = 900;
        logits[usize::from(b'z')] = 800;
        let decode = DecodeConfig {
            ascii_lower_only: true,
            ..DecodeConfig::greedy()
        };

        let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
            .expect("ascii lower decode");

        assert_eq!(token, b'z');
    }

    #[test]
    fn max_repeat_run_decode_breaks_greedy_loop() {
        let mut logits = [0_i32; BYTE_VOCAB];
        let probabilities = [1_i16; BYTE_VOCAB];
        logits[usize::from(b'a')] = 1000;
        logits[usize::from(b'b')] = 900;
        let decode = DecodeConfig {
            max_repeat_run: 3,
            ..DecodeConfig::greedy()
        };

        let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"aaa")
            .expect("run-capped decode");

        assert_eq!(token, b'b');
    }

    #[test]
    fn strict_adjacency_decode_rejects_unseen_successors() {
        let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
        let mut logits = [0_i32; BYTE_VOCAB];
        let probabilities = [1_i16; BYTE_VOCAB];
        logits[usize::from(b'z')] = 1000;
        logits[usize::from(b'b')] = 900;
        let decode = DecodeConfig {
            strict_adjacency: true,
            ..DecodeConfig::greedy()
        };

        let selection = select_byte_from_row_with_priors(
            &logits,
            &probabilities,
            decode,
            0,
            b"a",
            Some(&priors),
        )
        .expect("strict adjacency decode");

        assert_eq!(selection.token, b'b');
        assert_eq!(selection.candidate_count, 1);
        assert_eq!(selection.rejected_candidates.adjacency, BYTE_VOCAB - 1);
    }

    #[test]
    fn corpus_prior_can_rerank_greedy_decode() {
        let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
        let mut logits = [0_i32; BYTE_VOCAB];
        let probabilities = [1_i16; BYTE_VOCAB];
        logits[usize::from(b'z')] = 1000;
        logits[usize::from(b'b')] = 900;
        let decode = DecodeConfig {
            corpus_prior: true,
            corpus_prior_logit_shift: 7,
            ..DecodeConfig::greedy()
        };

        let selection = select_byte_from_row_with_priors(
            &logits,
            &probabilities,
            decode,
            0,
            b"a",
            Some(&priors),
        )
        .expect("corpus prior decode");

        assert_eq!(selection.token, b'b');
        assert_eq!(selection.candidate_count, BYTE_VOCAB);
        assert_eq!(selection.rejected_candidates.adjacency, 0);
    }

    #[test]
    fn corpus_prior_decode_requires_priors() {
        let logits = [0_i32; BYTE_VOCAB];
        let probabilities = [1_i16; BYTE_VOCAB];
        let decode = DecodeConfig {
            corpus_prior: true,
            ..DecodeConfig::greedy()
        };

        assert!(
            select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", None)
                .is_err()
        );
    }

    #[test]
    fn mini_transformer_embedding_sequence_includes_trainable_position_embedding() {
        let mut embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let position_embeddings = initial_mini_transformer_position_embeddings(2);
        let row_start = usize::from(b'a') * MINI_TRANSFORMER_D_MODEL;
        embeddings[row_start..row_start + MINI_TRANSFORMER_D_MODEL].fill(256);

        let sequence = mini_transformer_embedding_sequence_with_position_policy_q15(
            &embeddings,
            &position_embeddings,
            b"aa",
            MiniTransformerPositionPolicy::LearnedAbsolute,
        )
        .expect("sequence");
        let first = &sequence[..MINI_TRANSFORMER_D_MODEL];
        let second = &sequence[MINI_TRANSFORMER_D_MODEL..2 * MINI_TRANSFORMER_D_MODEL];

        assert_ne!(first, second);
        assert!(sequence.iter().all(|&value| (-768..=1280).contains(&value)));
    }

    #[test]
    fn mini_transformer_nope_embedding_sequence_skips_position_embedding() {
        let mut embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let position_embeddings = initial_mini_transformer_position_embeddings(2);
        let row_start = usize::from(b'a') * MINI_TRANSFORMER_D_MODEL;
        embeddings[row_start..row_start + MINI_TRANSFORMER_D_MODEL].fill(256);

        let sequence = mini_transformer_embedding_sequence_with_position_policy_q15(
            &embeddings,
            &position_embeddings,
            b"aa",
            MiniTransformerPositionPolicy::Nope,
        )
        .expect("sequence");
        let first = &sequence[..MINI_TRANSFORMER_D_MODEL];
        let second = &sequence[MINI_TRANSFORMER_D_MODEL..2 * MINI_TRANSFORMER_D_MODEL];

        assert_eq!(first, second);
        assert!(sequence.iter().all(|&value| value == 256));
    }

    #[cfg(not(feature = "mini-calibrated"))]
    #[test]
    fn mini_transformer_mlp_training_updates_head_mlp_and_attention() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 2,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(64),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 18,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("mini train");

        assert_eq!(trace.token_count, tokens.len());
        assert!(trace.windows > 0);
        assert!(trace.updates > 0);
        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert_ne!(trace.initial_model_hash, trace.final_model_hash);
        assert_ne!(trace.initial_embedding_hash, trace.final_embedding_hash);
        assert_ne!(trace.initial_output_head_hash, trace.final_output_head_hash);
        assert_ne!(trace.initial_mlp_hash, trace.final_mlp_hash);
        assert_ne!(trace.initial_attention_hash, trace.final_attention_hash);
        assert_eq!(trace.output_head_saturation_count, 0);
        assert!(trace.output_head_delta_l1 > 0);
        assert!(trace.mlp_delta_l1 > 0);
        assert!(trace.embedding_delta_l1 > 0);
        assert!(trace.attention_delta_l1 > 0);
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.mlp_hash_before != step.mlp_hash_after)
        );
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.attention_hash_before != step.attention_hash_after)
        );
    }

    #[test]
    fn mini_transformer_stacked_serial_training_updates_lower_layer() {
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        assert_eq!(model.transformer_layers(), 2);
        let context = b"To b";
        let cache = mini_transformer_forward_for_attention_and_position(
            &model,
            context,
            MiniTransformerAttentionKind::Base2Softmax,
            MiniTransformerPositionPolicy::LearnedAbsolute,
        )
        .expect("stacked forward");
        let first_attention_range = model
            .attention_weight_range(0)
            .expect("first attention range");
        let first_down_range = model.mlp_down_weight_range(0).expect("first down range");
        let initial_first_o_hash = hash_i8_slice(&model.o_weights[first_attention_range.clone()]);
        let initial_first_down_hash = hash_i8_slice(&model.down_weights[first_down_range.clone()]);
        let mut workspace =
            MiniTransformerHostTrainCoreWorkspaceBuffers::new(context.len()).expect("workspace");
        let mut grad_block_output = vec![0_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
        for (index, gradient) in grad_block_output.iter_mut().enumerate() {
            *gradient = if index.is_multiple_of(3) { 2048 } else { -1024 };
        }

        let update = mini_transformer_block_backward_update_i8_checked(
            &cache.layers[0],
            &grad_block_output,
            &mut model,
            0,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: context.len(),
                stride: 1,
                window_offset: 0,
                max_windows: Some(1),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                learning_rate: 2,
                output_learning_rate_shift: 16,
                mlp_learning_rate_shift: 10,
                embedding_learning_rate_shift: 12,
                attention_learning_rate_shift: 10,
                attention_q_learning_rate_shift: 10,
                attention_qk_learning_rate_shift: 10,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
            &mut workspace,
        )
        .expect("lower block backward");

        assert!(update.mlp_update.weight_delta_l1().unwrap_or(0) > 0);
        assert!(update.attention_update.weight_delta_l1 > 0);
        assert_ne!(
            hash_i8_slice(&model.down_weights[first_down_range]),
            initial_first_down_hash
        );
        assert_ne!(
            hash_i8_slice(&model.o_weights[first_attention_range]),
            initial_first_o_hash
        );
        assert_eq!(
            update.grad_input.len(),
            context.len() * MINI_TRANSFORMER_D_MODEL
        );
    }

    #[test]
    fn linear_attention_backward_produces_qkv_gradients() {
        let seq_len = 2;
        let total = seq_len * MINI_TRANSFORMER_D_MODEL;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut grad_context = vec![0_i16; total];
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            q[dim] = 256 + dim as i16 * 8;
            q[MINI_TRANSFORMER_D_MODEL + dim] = -128 + dim as i16 * 4;
            k[dim] = -192 + dim as i16 * 6;
            k[MINI_TRANSFORMER_D_MODEL + dim] = 224 - dim as i16 * 5;
            v[dim] = 8192 - dim as i16 * 16;
            v[MINI_TRANSFORMER_D_MODEL + dim] = -6144 + dim as i16 * 12;
            grad_context[dim] = 4096 + dim as i16 * 8;
            grad_context[MINI_TRANSFORMER_D_MODEL + dim] = -3072 + dim as i16 * 6;
        }

        let (grad_q, grad_k, grad_v) =
            mini_transformer_linear_attention_qkv_gradients_q15(seq_len, &q, &k, &v, &grad_context)
                .expect("linear gradients");

        assert_eq!(grad_q.len(), total);
        assert_eq!(grad_k.len(), total);
        assert_eq!(grad_v.len(), total);
        assert!(grad_q.iter().any(|&value| value != 0));
        assert!(grad_k.iter().any(|&value| value != 0));
        assert!(grad_v.iter().any(|&value| value != 0));
    }

    #[cfg(not(feature = "mini-calibrated"))]
    #[test]
    fn mini_transformer_mlp_training_can_use_linear_attention() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(16),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 13,
                attention_qk_learning_rate_shift: 16,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("linear mini train");

        assert!(trace.updates > 0);
        assert_eq!(trace.final_invalid_forward_count, 0);
        assert!(trace.attention_delta_l1 > 0);
        assert_ne!(trace.initial_attention_hash, trace.final_attention_hash);
        let line = trace.to_json_line();
        assert!(line.contains("\"attention_kind\":\"linear\""));
        assert!(line.contains(
            "\"attention_backward\":\"linear_numerator_straight_through_denominator_constant\""
        ));
        assert!(line.contains("\"attention_q_learning_rate_shift\":13"));
    }

    #[test]
    fn mini_transformer_adaptive_attention_shifts_are_traced() {
        let tokens = b"to be or not to be to be or not to be ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 2,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 22,
                attention_qk_learning_rate_shift: 22,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: true,
                adaptive_holographic_shifts: true,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("adaptive train");

        assert!(trace.adaptive_holographic_update_count > 0);
        assert!(
            trace.adaptive_holographic_update_count
                >= trace.adaptive_attention_holographic_update_count
        );
        assert!(trace.adaptive_attention_holographic_update_count > 0);
        assert!(trace.final_output_learning_rate_shift <= MAX_RIGHT_SHIFT);
        assert!(trace.final_mlp_learning_rate_shift <= MAX_RIGHT_SHIFT);
        assert!(trace.final_embedding_learning_rate_shift <= MAX_RIGHT_SHIFT);
        assert!(trace.final_attention_q_learning_rate_shift <= MAX_RIGHT_SHIFT);
        assert!(trace.final_attention_qk_learning_rate_shift <= MAX_RIGHT_SHIFT);
        assert!(trace.final_attention_learning_rate_shift <= MAX_RIGHT_SHIFT);
        let line = trace.to_json_line();
        assert!(line.contains("\"adaptive_attention_shifts\":true"));
        assert!(line.contains("\"adaptive_holographic_shifts\":true"));
        assert!(line.contains("\"adaptive_holographic_update_count\":"));
        assert!(line.contains("\"adaptive_holographic_meta_dim\":8"));
        assert!(line.contains("\"adaptive_holographic_action_count\":5"));
        assert!(line.contains("\"adaptive_attention_holographic_update_count\":"));
        assert!(line.contains("\"final_output_learning_rate_shift\":"));
        assert!(line.contains("\"final_attention_q_learning_rate_shift\":"));
    }

    #[test]
    fn mini_transformer_adaptive_holographic_shifts_enable_controller() {
        let tokens = b"to be or not to be to be or not to be ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 2,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 22,
                attention_qk_learning_rate_shift: 22,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: true,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("holographic adaptive train");

        assert!(trace.adaptive_holographic_update_count > 0);
        assert!(trace.adaptive_attention_holographic_update_count > 0);
        let line = trace.to_json_line();
        assert!(line.contains("\"adaptive_attention_shifts\":false"));
        assert!(line.contains("\"adaptive_holographic_shifts\":true"));
        assert!(line.contains("\"adaptive_holographic_meta_dim\":8"));
        assert!(line.contains("\"adaptive_holographic_hash\":"));
    }

    #[test]
    fn mini_transformer_holographic_memory_binds_previous_state_to_next_teacher() {
        let mut memory = IntegerHolographicShiftMemory::new();
        let mut previous_state = None;
        let state_a = [i16::MAX, 1024, 0, 512, 0, 0, 256, -512];
        let state_b = [i16::MAX, 2048, 256, 768, 0, 0, 512, 0];

        mini_transformer_holo_remember_lagged(&mut memory, &mut previous_state, state_a, -1);
        assert_eq!(memory.update_count, 0);
        assert_eq!(previous_state, Some(state_a));

        mini_transformer_holo_remember_lagged(&mut memory, &mut previous_state, state_b, 1);
        assert_eq!(memory.update_count, 1);
        assert_eq!(previous_state, Some(state_b));
        assert_eq!(memory.retrieve_delta(&state_a), Some(1));
    }

    #[test]
    fn mini_transformer_holographic_memory_can_act_when_teacher_is_silent() {
        assert_eq!(mini_transformer_holo_safety_delta(0, -2, false), -1);
        assert_eq!(mini_transformer_holo_safety_delta(0, -1, false), -1);
        assert_eq!(mini_transformer_holo_safety_delta(0, 0, false), 0);
        assert_eq!(mini_transformer_holo_safety_delta(0, 1, false), 1);
        assert_eq!(mini_transformer_holo_safety_delta(0, 2, false), 1);
        assert_eq!(mini_transformer_holo_safety_delta(1, -1, true), 1);
        assert_eq!(mini_transformer_holo_safety_delta(-1, 1, true), -1);
        assert_eq!(mini_transformer_holo_safety_delta(1, -1, false), 0);
        assert_eq!(mini_transformer_holo_safety_delta(-1, 1, false), 0);
    }

    #[test]
    fn mini_transformer_holographic_authority_requires_history_and_cooldown() {
        let mut last_adjust_batch = None;
        assert_eq!(
            mini_transformer_holo_authorized_delta(
                -1,
                0,
                MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES - 1,
                1,
                &mut last_adjust_batch,
            ),
            0
        );
        assert_eq!(last_adjust_batch, None);
        assert_eq!(
            mini_transformer_holo_authorized_delta(
                -1,
                0,
                MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES,
                8,
                &mut last_adjust_batch,
            ),
            -1
        );
        assert_eq!(last_adjust_batch, Some(8));
        assert_eq!(
            mini_transformer_holo_authorized_delta(
                1,
                1,
                0,
                8 + MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES - 1,
                &mut last_adjust_batch,
            ),
            0
        );
        assert_eq!(
            mini_transformer_holo_authorized_delta(
                1,
                1,
                0,
                8 + MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES,
                &mut last_adjust_batch,
            ),
            1
        );
    }

    #[test]
    fn mini_transformer_adaptive_rule_shifts_emit_events() {
        let tokens = b"to be or not to be to be or not to be ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 2,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 22,
                attention_qk_learning_rate_shift: 22,
                adaptive_rule_shifts: true,
                adaptive_rule_interval_batches: 1,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("rule adaptive train");

        assert!(trace.adaptive_rule_update_count > 0);
        assert!(trace.adaptive_rule_shift_adjustment_count > 0);
        assert_eq!(trace.adaptive_holographic_update_count, 0);
        assert_eq!(trace.adaptive_holographic_shift_adjustment_count, 0);
        assert!(!trace.adaptive_shift_events.is_empty());
        let line = trace.to_json_line();
        assert!(line.contains("\"adaptive_rule_shifts\":true"));
        assert!(line.contains("\"adaptive_rule_interval_batches\":1"));
        assert!(line.contains("\"adaptive_rule_shift_adjustment_count\":"));
        assert!(line.contains("\"adaptive_shift_events\":["));
        assert!(line.contains("\"component\":\""));
        assert!(line.contains("\"reason\":\""));
    }

    #[test]
    fn mini_transformer_rule_saturation_is_window_gated() {
        let interval = 4;
        let weight_count = 512;
        let quiet_stats = LinearWeightUpdateStats {
            gradient_saturation_count: 0,
            zero_delta_count: 0,
            weight_delta_l1: 1,
        };
        let sparse_saturation = LinearWeightUpdateStats {
            gradient_saturation_count: 1,
            zero_delta_count: 0,
            weight_delta_l1: 1,
        };

        let mut sparse_window = MiniTransformerRuleShiftWindow::new();
        sparse_window.observe_accepted(sparse_saturation);
        assert_eq!(
            mini_transformer_rule_generic_delta(sparse_window, weight_count, interval),
            None
        );
        assert!(!mini_transformer_rule_should_reset(sparse_window, interval));
        for _ in 1..interval {
            sparse_window.observe_accepted(quiet_stats);
        }
        assert_eq!(
            mini_transformer_rule_generic_delta(sparse_window, weight_count, interval),
            None
        );
        assert!(mini_transformer_rule_should_reset(sparse_window, interval));

        let mut pressure_window = MiniTransformerRuleShiftWindow::new();
        for _ in 0..interval {
            pressure_window.observe_accepted(sparse_saturation);
        }
        assert_eq!(
            mini_transformer_rule_generic_delta(pressure_window, weight_count, interval),
            Some((1, "saturation"))
        );
    }

    #[test]
    fn mini_transformer_qk_rule_prioritizes_dead_gradient_over_saturation() {
        let interval = 4;
        let mut q_window = MiniTransformerRuleShiftWindow::new();
        let mut k_window = MiniTransformerRuleShiftWindow::new();
        let dead_saturating_q = LinearWeightUpdateStats {
            gradient_saturation_count: 1024,
            zero_delta_count: mini_transformer_attention_projection_weight_count(),
            weight_delta_l1: 0,
        };
        let moving_k = LinearWeightUpdateStats {
            gradient_saturation_count: 0,
            zero_delta_count: 0,
            weight_delta_l1: 100_000,
        };

        for _ in 0..interval {
            q_window.observe_accepted(dead_saturating_q);
            k_window.observe_accepted(moving_k);
        }

        assert_eq!(
            mini_transformer_rule_q_delta(q_window, k_window, interval),
            Some((-1, "zero_delta"))
        );
    }

    #[test]
    fn mini_transformer_k_rule_prioritizes_dead_gradient_over_saturation() {
        let interval = 4;
        let mut k_window = MiniTransformerRuleShiftWindow::new();
        let mut q_window = MiniTransformerRuleShiftWindow::new();
        let dead_saturating_k = LinearWeightUpdateStats {
            gradient_saturation_count: 1024,
            zero_delta_count: mini_transformer_attention_projection_weight_count(),
            weight_delta_l1: 0,
        };
        let moving_q = LinearWeightUpdateStats {
            gradient_saturation_count: 0,
            zero_delta_count: 0,
            weight_delta_l1: 100_000,
        };

        for _ in 0..interval {
            k_window.observe_accepted(dead_saturating_k);
            q_window.observe_accepted(moving_q);
        }

        assert_eq!(
            mini_transformer_rule_k_delta(k_window, q_window, interval),
            Some((-1, "zero_delta"))
        );
    }

    #[test]
    fn mini_transformer_mlp_training_trace_is_byte_stable() {
        let tokens = b"abababababab";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(6),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let left = run_mini_transformer_mlp_training(tokens, config)
            .expect("left")
            .to_json_line();
        let right = run_mini_transformer_mlp_training(tokens, config)
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_mini_transformer_mlp_trace.v1\""));
        assert!(left.contains("\"attention\":\"updates_q_k_v_o_i8\""));
        assert!(left.contains("\"rejected_window_count\":"));
        assert!(left.contains("\"final_invalid_forward_count\":"));
        assert!(left.contains(
            "\"trained_component\":\"embedding_i16_plus_output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8\""
        ));
    }

    #[test]
    fn mini_transformer_binary_trace_has_fixed_step_records() {
        let tokens = b"abababababab";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(6),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let trace = run_mini_transformer_mlp_training(tokens, config).expect("trace");
        let binary = trace.to_binary_trace_v1();
        let final_offset = 16 + trace.steps.len() * 32;

        assert_eq!(&binary[..4], MINI_TRANSFORMER_BINARY_TRACE_MAGIC);
        assert_eq!(binary[4], MINI_TRANSFORMER_BINARY_TRACE_VERSION);
        assert_eq!(binary[5], MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID);
        assert_eq!(binary[16], MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE);
        assert_eq!(
            binary[final_offset],
            MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY
        );
        assert_eq!(binary[final_offset + 1], 0);
    }

    #[test]
    fn mini_transformer_streamed_binary_trace_matches_buffered_trace() {
        let tokens = b"to be or not to be ";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 2,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let mut streamed = Vec::new();
        let buffered = {
            let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
            let mut writer = MiniTransformerBinaryTraceWriter::new(&mut streamed);
            let run =
                run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace(
                    tokens,
                    config,
                    model,
                    0,
                    MiniTransformerTraceDetail::Summary,
                    |_| Ok(()),
                    |record| writer.write_record(record).map_err(|_| TrainError::TraceWrite),
                )
                .expect("streamed binary trace");
            run.trace.to_binary_trace_v1()
        };

        assert_eq!(streamed, buffered);
    }

    #[test]
    fn mini_transformer_swarm_trains_interleaved_worker_shards() {
        let tokens = b"to be or not to be to be or not to be ";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let run = run_mini_transformer_mlp_swarm_training(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: 2,
                trace_detail: MiniTransformerTraceDetail::None,
            },
        )
        .expect("swarm training");

        assert_eq!(run.trace.worker_count, 2);
        assert_eq!(run.trace.workers.len(), 2);
        assert_eq!(run.trace.workers[0].window_offset, 0);
        assert_eq!(run.trace.workers[1].window_offset, 1);
        assert_eq!(run.trace.workers[0].stride, 2);
        assert_eq!(run.trace.workers[1].stride, 2);
        assert_eq!(run.trace.workers[0].max_windows, Some(4));
        assert_eq!(run.trace.workers[1].max_windows, Some(4));
        assert_eq!(run.trace.final_model_hash, run.model.model_hash());
        assert!(
            run.trace
                .workers
                .iter()
                .any(|worker| worker.worker_index == run.trace.best_worker_index)
        );
        assert!(
            run.trace
                .to_json_line()
                .contains("\"schema\":\"nsrl.training_mini_transformer_swarm_trace.v1\"")
        );
    }

    #[test]
    fn mini_transformer_swarm_worker_artifacts_assemble_to_local_swarm() {
        let tokens = b"to be or not to be to be or not to be ";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let base_model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        let local = run_mini_transformer_mlp_swarm_training_from_model(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: 2,
                trace_detail: MiniTransformerTraceDetail::None,
            },
            base_model.clone(),
        )
        .expect("local swarm");
        let artifacts = (0..2)
            .map(|worker_index| {
                let run = run_mini_transformer_mlp_swarm_worker_from_model_with_progress(
                    tokens,
                    config,
                    worker_index,
                    2,
                    base_model.clone(),
                    0,
                    MiniTransformerTraceDetail::None,
                    |_| Ok(()),
                )
                .expect("worker");
                let bytes = run.artifact.try_to_bytes().expect("worker bytes");
                MiniTransformerMlpSwarmWorkerArtifact::from_bytes(&bytes).expect("worker artifact")
            })
            .collect::<Vec<_>>();
        let assembled = assemble_mini_transformer_mlp_swarm_worker_artifacts(
            tokens,
            config,
            &base_model,
            artifacts,
        )
        .expect("assembled swarm");

        assert_eq!(assembled.trace, local.trace);
        assert_eq!(assembled.model, local.model);
        assert_eq!(assembled.swarm_model, local.swarm_model);
        assert!(
            assembled
                .trace
                .to_json_line()
                .contains("\"schema\":\"nsrl.training_mini_transformer_swarm_trace.v1\"")
        );
    }

    #[test]
    fn mini_transformer_swarm_trace_is_byte_stable() {
        let tokens = b"abababababababab";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(6),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let swarm_config = MiniTransformerMlpSwarmTrainConfig {
            workers: 3,
            trace_detail: MiniTransformerTraceDetail::None,
        };
        let left = run_mini_transformer_mlp_swarm_training(tokens, config, swarm_config)
            .expect("left")
            .trace
            .to_json_line();
        let right = run_mini_transformer_mlp_swarm_training(tokens, config, swarm_config)
            .expect("right")
            .trace
            .to_json_line();

        assert_eq!(left, right);
    }

    #[test]
    fn mini_transformer_swarm_scaling_benchmark_sweeps_worker_counts() {
        let tokens = b"abababababababab";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(6),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let trace = run_mini_transformer_mlp_swarm_scaling_benchmark(
            tokens,
            config,
            3,
            MiniTransformerTraceDetail::None,
        )
        .expect("scaling benchmark");

        assert_eq!(trace.worker_counts, vec![1, 2, 3]);
        assert_eq!(trace.runs.len(), 3);
        assert_eq!(trace.runs[0].requested_worker_count, 1);
        assert_eq!(trace.runs[0].effective_worker_count, 1);
        assert_eq!(trace.runs[0].speedup_per_mille, 1000);
        assert!(trace.runs.iter().all(|run| {
            run.effective_worker_count > 0
                && run.effective_worker_count <= run.requested_worker_count
                && run.examined_windows > 0
        }));

        let json = trace.to_json_line();
        assert!(
            json.contains("\"schema\":\"nsrl.training_mini_transformer_swarm_scaling_trace.v1\"")
        );
        assert!(json.contains("\"worker_counts\":[1,2,3]"));
        assert!(json.contains("\"speedup_per_mille\""));
    }

    #[test]
    fn mini_transformer_swarm_model_roundtrips_and_generates() {
        let tokens = b"to be or not to be to be or not to be ";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let run = run_mini_transformer_mlp_swarm_training(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: 2,
                trace_detail: MiniTransformerTraceDetail::None,
            },
        )
        .expect("swarm training");
        let bytes = run.swarm_model.try_to_bytes().expect("swarm bytes");
        let decoded = MiniTransformerMlpSwarmModel::from_bytes(&bytes).expect("swarm model");

        assert_eq!(decoded, run.swarm_model);
        assert_eq!(decoded.worker_count(), 2);
        assert_eq!(decoded.model_hash(), run.swarm_model.model_hash());
        let manifest = decoded.to_expert_manifest().expect("swarm manifest");

        assert_eq!(manifest.artifact_byte_count, bytes.len());
        assert_eq!(manifest.model_hash, decoded.model_hash());
        assert_eq!(manifest.worker_count, 2);
        assert_eq!(manifest.worker_model_hashes.len(), 2);
        assert_eq!(manifest.worker_parameter_bytes.len(), 2);
        assert_eq!(
            manifest.parameter_bytes,
            manifest.worker_parameter_bytes.iter().sum::<usize>()
        );
        let manifest_json = manifest.to_json_line();
        assert!(
            manifest_json.contains("\"schema\":\"nsrl.mini_transformer_swarm_expert_manifest.v1\"")
        );
        assert!(manifest_json.contains("\"supported_compositions\":[\"average_logits\",\"confidence_weighted\",\"confidence_router\"]"));
        assert!(manifest_json.contains("\"artifact\":{\"format\":\"nsrlswarm\""));
        let mut oversized_manifest = manifest.clone();
        oversized_manifest.parameter_bytes = manifest.parameter_bytes.saturating_add(1);
        let route = route_mini_transformer_swarm_experts(
            &[
                MiniTransformerSwarmRouteCandidate {
                    expert_id: String::from("fit.nsrlswarm"),
                    manifest: manifest.clone(),
                },
                MiniTransformerSwarmRouteCandidate {
                    expert_id: String::from("too-large.nsrlswarm"),
                    manifest: oversized_manifest,
                },
            ],
            MiniTransformerSwarmRouteConfig {
                required_capabilities: vec![
                    String::from("byte_generation"),
                    String::from("integer_q15"),
                ],
                max_artifact_bytes: Some(bytes.len()),
                max_parameter_bytes: Some(manifest.parameter_bytes),
                active_expert_limit: 1,
                prompt_affinity: false,
                prompt_affinity_max_windows: 32,
            },
            b"to be",
        )
        .expect("swarm route");
        assert_eq!(route.selected_expert_indices, vec![0]);
        assert!(route.candidates[0].accepted);
        assert_eq!(
            route.candidates[0].matched_capabilities,
            vec![String::from("byte_generation"), String::from("integer_q15")]
        );
        assert!(route.candidates[0].missing_capabilities.is_empty());
        assert!(!route.candidates[1].accepted);
        assert_eq!(
            route.candidates[1].reject_reason,
            "parameter_budget_exceeded"
        );
        let route_json = route.to_json_line();
        assert!(route_json.contains("\"schema\":\"nsrl.mini_transformer_swarm_route_trace.v1\""));
        assert!(route_json.contains("\"selected_expert_indices\":[0]"));
        let prompt_affinity_route = route_mini_transformer_swarm_expert_models(
            &[
                MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: String::from("left.nsrlswarm"),
                    model: decoded.clone(),
                },
                MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: String::from("right.nsrlswarm"),
                    model: decoded.clone(),
                },
            ],
            MiniTransformerSwarmRouteConfig {
                required_capabilities: vec![String::from("byte_generation")],
                max_artifact_bytes: Some(bytes.len()),
                max_parameter_bytes: None,
                active_expert_limit: 1,
                prompt_affinity: true,
                prompt_affinity_max_windows: 4,
            },
            b"to be",
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
            MiniTransformerSwarmComposition::ConfidenceRouter,
        )
        .expect("prompt-affinity route");
        assert_eq!(prompt_affinity_route.selected_expert_indices, vec![0]);
        assert!(
            prompt_affinity_route
                .candidates
                .iter()
                .all(|candidate| candidate.prompt_eval_windows > 0
                    && candidate.prompt_probability_error_q15.is_some())
        );
        assert!(
            route_mini_transformer_swarm_experts(
                &[MiniTransformerSwarmRouteCandidate {
                    expert_id: String::from("fit.nsrlswarm"),
                    manifest,
                }],
                MiniTransformerSwarmRouteConfig {
                    required_capabilities: vec![String::from("lexeme_generation")],
                    max_artifact_bytes: None,
                    max_parameter_bytes: None,
                    active_expert_limit: 1,
                    prompt_affinity: false,
                    prompt_affinity_max_windows: 32,
                },
                b"to be",
            )
            .is_err()
        );
        let routed_generation = generate_routed_mini_transformer_swarm_experts(
            &[
                MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: String::from("left.nsrlswarm"),
                    model: decoded.clone(),
                },
                MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: String::from("right.nsrlswarm"),
                    model: decoded.clone(),
                },
            ],
            MiniTransformerSwarmRouteConfig {
                required_capabilities: vec![String::from("byte_generation")],
                max_artifact_bytes: Some(bytes.len()),
                max_parameter_bytes: None,
                active_expert_limit: 2,
                prompt_affinity: true,
                prompt_affinity_max_windows: 4,
            },
            b"to be",
            ByteGenerationConfig::greedy(2),
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
            MiniTransformerSwarmComposition::ConfidenceRouter,
            None,
        )
        .expect("routed swarm generation");
        assert_eq!(routed_generation.route.selected_expert_indices, vec![0, 1]);
        assert!(routed_generation.route.candidates.iter().all(|candidate| {
            candidate.prompt_eval_windows > 0 && candidate.prompt_probability_error_q15.is_some()
        }));
        assert_eq!(routed_generation.selected_expert_ids.len(), 2);
        assert_eq!(routed_generation.active_worker_count, 4);
        assert_eq!(
            routed_generation.generation.composition,
            MiniTransformerSwarmComposition::ConfidenceRouter
        );
        assert_eq!(routed_generation.generation.generated_bytes.len(), 2);
        assert!(
            routed_generation
                .to_json_line()
                .contains("\"schema\":\"nsrl.mini_transformer_swarm_routed_generation_trace.v1\"")
        );

        let generation =
            generate_mini_transformer_swarm_with_attention_kind_position_policy_and_priors(
                &decoded,
                b"to be",
                ByteGenerationConfig::greedy(4),
                MiniTransformerAttentionKind::Linear,
                MiniTransformerPositionPolicy::Nope,
                None,
            )
            .expect("swarm generation");

        assert_eq!(generation.worker_count, 2);
        assert_eq!(generation.swarm_model_hash, decoded.model_hash());
        assert_eq!(
            generation.composition,
            MiniTransformerSwarmComposition::AverageLogits
        );
        assert_eq!(generation.generated_bytes.len(), 4);
        assert!(
            generation
                .to_json_line()
                .contains("\"schema\":\"nsrl.mini_transformer_swarm_generation_trace.v1\"")
        );

        let weighted_generation =
            generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
                &decoded,
                b"to be",
                ByteGenerationConfig::greedy(2),
                MiniTransformerAttentionKind::Linear,
                MiniTransformerPositionPolicy::Nope,
                MiniTransformerSwarmComposition::ConfidenceWeighted,
                None,
            )
            .expect("weighted swarm generation");
        let router_generation =
            generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
                &decoded,
                b"to be",
                ByteGenerationConfig::greedy(2),
                MiniTransformerAttentionKind::Linear,
                MiniTransformerPositionPolicy::Nope,
                MiniTransformerSwarmComposition::ConfidenceRouter,
                None,
            )
            .expect("router swarm generation");

        assert_eq!(
            weighted_generation.composition,
            MiniTransformerSwarmComposition::ConfidenceWeighted
        );
        assert_eq!(
            router_generation.composition,
            MiniTransformerSwarmComposition::ConfidenceRouter
        );
        assert!(
            router_generation
                .to_json_line()
                .contains("\"composition\":\"confidence_router\"")
        );
    }

    #[cfg(not(feature = "mini-calibrated"))]
    struct MiniTransformerTrainCoreWorkspaceBuffers {
        embedding_output: Vec<i16>,
        attention_norm: Vec<i16>,
        attention_q: Vec<i16>,
        attention_k: Vec<i16>,
        attention_v: Vec<i16>,
        attention_context: Vec<i16>,
        attention_output: Vec<i16>,
        attention_residual: Vec<i16>,
        attention_state_kv: Vec<i64>,
        attention_key_sums: Vec<i64>,
        mlp_norm: Vec<i16>,
        mlp_up: Vec<i16>,
        mlp_gate: Vec<i16>,
        mlp_gated: Vec<i16>,
        mlp_output: Vec<i16>,
        block_output: Vec<i16>,
        logits_q8: Vec<i32>,
        probabilities_q15: Vec<i16>,
        grad_output_q15: Vec<i16>,
        output_scaled_grad: Vec<i32>,
        grad_last_features: Vec<i16>,
        grad_mlp_output: Vec<i16>,
        grad_mlp_input: Vec<i16>,
        mlp_scaled_grad: Vec<i32>,
        mlp_input_grad_gated: Vec<i16>,
        mlp_input_grad_up: Vec<i16>,
        mlp_input_grad_gate: Vec<i16>,
        mlp_input_grad_up_input: Vec<i16>,
        mlp_input_grad_gate_input: Vec<i16>,
        mlp_update_grad_gated: Vec<i16>,
        mlp_update_grad_up: Vec<i16>,
        mlp_update_grad_gate: Vec<i16>,
        grad_attention_output: Vec<i16>,
        grad_attention_context: Vec<i16>,
        attention_scaled_grad: Vec<i32>,
        linear_prefix_states: Vec<i64>,
        linear_denominators: Vec<i64>,
        linear_grad_state_q15: Vec<i64>,
        linear_grad_q_acc: Vec<i64>,
        linear_grad_k_acc: Vec<i64>,
        linear_grad_v_acc: Vec<i64>,
        grad_attention_q: Vec<i16>,
        grad_attention_k: Vec<i16>,
        grad_attention_v: Vec<i16>,
        grad_attention_norm_input: Vec<i16>,
        grad_embedding_output: Vec<i16>,
    }

    #[cfg(not(feature = "mini-calibrated"))]
    impl MiniTransformerTrainCoreWorkspaceBuffers {
        fn new(seq_len: usize) -> Self {
            assert_eq!(
                nsrl_train_core::MINI_TRANSFORMER_D_MODEL,
                MINI_TRANSFORMER_D_MODEL
            );
            assert_eq!(
                nsrl_train_core::MINI_TRANSFORMER_HEADS,
                MINI_TRANSFORMER_HEADS
            );
            assert_eq!(
                nsrl_train_core::MINI_TRANSFORMER_HIDDEN_DIM,
                MINI_TRANSFORMER_HIDDEN_DIM
            );
            assert_eq!(nsrl_train_core::BYTE_VOCAB, BYTE_VOCAB);

            let total = seq_len * MINI_TRANSFORMER_D_MODEL;
            let hidden_total = seq_len * MINI_TRANSFORMER_HIDDEN_DIM;
            let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
            let head_state_len = head_dim * head_dim;
            let state_len = MINI_TRANSFORMER_HEADS * head_state_len;
            let key_sum_len = MINI_TRANSFORMER_HEADS * head_dim;
            let prefix_len = seq_len * state_len;
            let denom_len = seq_len * MINI_TRANSFORMER_HEADS;
            let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

            Self {
                embedding_output: vec![0_i16; total],
                attention_norm: vec![0_i16; total],
                attention_q: vec![0_i16; total],
                attention_k: vec![0_i16; total],
                attention_v: vec![0_i16; total],
                attention_context: vec![0_i16; total],
                attention_output: vec![0_i16; total],
                attention_residual: vec![0_i16; total],
                attention_state_kv: vec![0_i64; state_len],
                attention_key_sums: vec![0_i64; key_sum_len],
                mlp_norm: vec![0_i16; total],
                mlp_up: vec![0_i16; hidden_total],
                mlp_gate: vec![0_i16; hidden_total],
                mlp_gated: vec![0_i16; hidden_total],
                mlp_output: vec![0_i16; total],
                block_output: vec![0_i16; total],
                logits_q8: vec![0_i32; BYTE_VOCAB],
                probabilities_q15: vec![0_i16; BYTE_VOCAB],
                grad_output_q15: vec![0_i16; BYTE_VOCAB],
                output_scaled_grad: vec![0_i32; BYTE_VOCAB],
                grad_last_features: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
                grad_mlp_output: vec![0_i16; total],
                grad_mlp_input: vec![0_i16; total],
                mlp_scaled_grad: vec![0_i32; scaled_len],
                mlp_input_grad_gated: vec![0_i16; hidden_total],
                mlp_input_grad_up: vec![0_i16; hidden_total],
                mlp_input_grad_gate: vec![0_i16; hidden_total],
                mlp_input_grad_up_input: vec![0_i16; total],
                mlp_input_grad_gate_input: vec![0_i16; total],
                mlp_update_grad_gated: vec![0_i16; hidden_total],
                mlp_update_grad_up: vec![0_i16; hidden_total],
                mlp_update_grad_gate: vec![0_i16; hidden_total],
                grad_attention_output: vec![0_i16; total],
                grad_attention_context: vec![0_i16; total],
                attention_scaled_grad: vec![0_i32; MINI_TRANSFORMER_D_MODEL],
                linear_prefix_states: vec![0_i64; prefix_len],
                linear_denominators: vec![0_i64; denom_len],
                linear_grad_state_q15: vec![0_i64; head_state_len],
                linear_grad_q_acc: vec![0_i64; total],
                linear_grad_k_acc: vec![0_i64; total],
                linear_grad_v_acc: vec![0_i64; total],
                grad_attention_q: vec![0_i16; total],
                grad_attention_k: vec![0_i16; total],
                grad_attention_v: vec![0_i16; total],
                grad_attention_norm_input: vec![0_i16; total],
                grad_embedding_output: vec![0_i16; total],
            }
        }

        fn as_workspace(&mut self) -> nsrl_train_core::MiniTransformerStepWorkspace<'_> {
            nsrl_train_core::MiniTransformerStepWorkspace {
                embedding_output: &mut self.embedding_output,
                attention_norm: &mut self.attention_norm,
                attention_q: &mut self.attention_q,
                attention_k: &mut self.attention_k,
                attention_v: &mut self.attention_v,
                attention_context: &mut self.attention_context,
                attention_output: &mut self.attention_output,
                attention_residual: &mut self.attention_residual,
                attention_state_kv: &mut self.attention_state_kv,
                attention_key_sums: &mut self.attention_key_sums,
                mlp_norm: &mut self.mlp_norm,
                mlp_up: &mut self.mlp_up,
                mlp_gate: &mut self.mlp_gate,
                mlp_gated: &mut self.mlp_gated,
                mlp_output: &mut self.mlp_output,
                block_output: &mut self.block_output,
                logits_q8: &mut self.logits_q8,
                probabilities_q15: &mut self.probabilities_q15,
                grad_output_q15: &mut self.grad_output_q15,
                output_scaled_grad: &mut self.output_scaled_grad,
                grad_last_features: &mut self.grad_last_features,
                grad_mlp_output: &mut self.grad_mlp_output,
                grad_mlp_input: &mut self.grad_mlp_input,
                mlp_scaled_grad: &mut self.mlp_scaled_grad,
                mlp_input_grad_gated: &mut self.mlp_input_grad_gated,
                mlp_input_grad_up: &mut self.mlp_input_grad_up,
                mlp_input_grad_gate: &mut self.mlp_input_grad_gate,
                mlp_input_grad_up_input: &mut self.mlp_input_grad_up_input,
                mlp_input_grad_gate_input: &mut self.mlp_input_grad_gate_input,
                mlp_update_grad_gated: &mut self.mlp_update_grad_gated,
                mlp_update_grad_up: &mut self.mlp_update_grad_up,
                mlp_update_grad_gate: &mut self.mlp_update_grad_gate,
                grad_attention_output: &mut self.grad_attention_output,
                grad_attention_context: &mut self.grad_attention_context,
                attention_scaled_grad: &mut self.attention_scaled_grad,
                linear_prefix_states: &mut self.linear_prefix_states,
                linear_denominators: &mut self.linear_denominators,
                linear_grad_state_q15: &mut self.linear_grad_state_q15,
                linear_grad_q_acc: &mut self.linear_grad_q_acc,
                linear_grad_k_acc: &mut self.linear_grad_k_acc,
                linear_grad_v_acc: &mut self.linear_grad_v_acc,
                grad_attention_q: &mut self.grad_attention_q,
                grad_attention_k: &mut self.grad_attention_k,
                grad_attention_v: &mut self.grad_attention_v,
                grad_attention_norm_input: &mut self.grad_attention_norm_input,
                grad_embedding_output: &mut self.grad_embedding_output,
            }
        }
    }

    #[cfg(not(feature = "mini-calibrated"))]
    #[test]
    fn mini_transformer_train_core_linear_nope_step_matches_std_single_window() {
        let tokens = b"To be";
        let seq_len = 4;
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len,
            stride: 1,
            window_offset: 0,
            max_windows: Some(1),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 17,
            embedding_learning_rate_shift: 13,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 16,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };
        let initial_model =
            MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(seq_len, 1)
                .expect("single-layer model");
        let std_run =
            run_mini_transformer_mlp_training_from_model(tokens, config, initial_model.clone())
                .expect("std training");
        assert_eq!(std_run.trace.updates, 1);
        assert_eq!(std_run.trace.rollback_count, 0);
        assert_eq!(std_run.trace.rejected_window_count, 0);

        let mut core_model = initial_model;
        let core_stats = {
            let mut model_slices = nsrl_train_core::MiniTransformerModelSlicesMut {
                embeddings: &mut core_model.embeddings,
                q_weights: &mut core_model.q_weights,
                k_weights: &mut core_model.k_weights,
                v_weights: &mut core_model.v_weights,
                o_weights: &mut core_model.o_weights,
                up_weights: &mut core_model.up_weights,
                gate_weights: &mut core_model.gate_weights,
                down_weights: &mut core_model.down_weights,
                output_weights: &mut core_model.output_weights,
            };
            let mut buffers = MiniTransformerTrainCoreWorkspaceBuffers::new(seq_len);
            let mut workspace = buffers.as_workspace();
            nsrl_train_core::mini_transformer_linear_nope_train_step(
                &mut model_slices,
                &tokens[..seq_len],
                tokens[seq_len],
                nsrl_train_core::MiniTransformerStepConfig {
                    seq_len,
                    learning_rate: config.learning_rate,
                    output_learning_rate_shift: config.output_learning_rate_shift,
                    mlp_learning_rate_shift: config.mlp_learning_rate_shift,
                    embedding_learning_rate_shift: config.embedding_learning_rate_shift,
                    attention_learning_rate_shift: config.attention_learning_rate_shift,
                    attention_q_learning_rate_shift: config.attention_q_learning_rate_shift,
                    attention_qk_learning_rate_shift: config.attention_qk_learning_rate_shift,
                },
                &mut workspace,
            )
            .expect("train core step")
        };
        let std_step = &std_run.trace.steps[0];
        assert_eq!(core_stats.predicted_before, std_step.predicted_token_before);
        assert_eq!(core_stats.predicted_after, std_step.predicted_token_after);
        assert_eq!(
            core_stats.output_head.gradient_saturation_count,
            std_step.output_head_saturation_count
        );
        assert_eq!(
            core_stats.output_head.zero_delta_count,
            std_step.output_head_zero_delta_count
        );
        assert_eq!(
            core_stats.output_head.weight_delta_l1,
            std_step.output_head_delta_l1
        );
        assert_eq!(
            core_stats.mlp.gradient_saturation_count(),
            std_step.mlp_saturation_count
        );
        assert_eq!(
            core_stats.mlp.zero_delta_count(),
            std_step.mlp_zero_delta_count
        );
        assert_eq!(core_stats.mlp.weight_delta_l1(), std_step.mlp_delta_l1);
        assert_eq!(
            core_stats.embedding.gradient_saturation_count,
            std_step.embedding_saturation_count
        );
        assert_eq!(
            core_stats.embedding.zero_delta_count,
            std_step.embedding_zero_delta_count
        );
        assert_eq!(
            core_stats.embedding.weight_delta_l1,
            std_step.embedding_delta_l1
        );
        assert_eq!(
            core_stats.attention.gradient_saturation_count(),
            std_step.attention_saturation_count
        );
        assert_eq!(
            core_stats.attention.zero_delta_count(),
            std_step.attention_zero_delta_count
        );
        assert_eq!(
            core_stats.attention.weight_delta_l1(),
            std_step.attention_delta_l1
        );
        assert_eq!(
            core_stats.attention.q.weight_delta_l1,
            std_step.attention_q_delta_l1
        );
        assert_eq!(
            core_stats.attention.k.weight_delta_l1,
            std_step.attention_k_delta_l1
        );
        assert_eq!(
            core_stats.attention.v.weight_delta_l1,
            std_step.attention_v_delta_l1
        );
        assert_eq!(
            core_stats.attention.o.weight_delta_l1,
            std_step.attention_o_delta_l1
        );
        assert_eq!(
            core_stats.residual_saturation_count,
            std_step.residual_saturation_count
        );
        assert_eq!(core_model.embeddings, std_run.model.embeddings);
        assert_eq!(
            core_model.position_embeddings,
            std_run.model.position_embeddings
        );
        assert_eq!(core_model.q_weights, std_run.model.q_weights);
        assert_eq!(core_model.k_weights, std_run.model.k_weights);
        assert_eq!(core_model.v_weights, std_run.model.v_weights);
        assert_eq!(core_model.o_weights, std_run.model.o_weights);
        assert_eq!(core_model.up_weights, std_run.model.up_weights);
        assert_eq!(core_model.gate_weights, std_run.model.gate_weights);
        assert_eq!(core_model.down_weights, std_run.model.down_weights);
        assert_eq!(core_model.output_weights, std_run.model.output_weights);
    }

    #[test]
    fn mini_transformer_nope_training_does_not_update_position_embeddings() {
        let tokens = b"To be or not to be, that is the question. To be or not to be. ";
        let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
            .expect("single-layer model");
        let initial_positions = initial_model.position_embeddings.clone();
        let initial_token_embedding_hash = hash_i16_slice(&initial_model.embeddings);
        let run = run_mini_transformer_mlp_training_from_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(16),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 18,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
            initial_model,
        )
        .expect("nope train");

        assert_eq!(run.model.position_embeddings, initial_positions);
        assert_ne!(
            hash_i16_slice(&run.model.embeddings),
            initial_token_embedding_hash
        );
        assert!(run.trace.to_json_line().contains("\"position\":\"nope\""));
    }

    #[test]
    fn mini_transformer_batch_windows_are_traced() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
            .expect("single-layer model");
        let trace = run_mini_transformer_mlp_training_from_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 4,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 18,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
            initial_model,
        )
        .expect("mini batch train")
        .trace;

        assert_eq!(trace.examined_windows, 8);
        assert_eq!(trace.accepted_batch_count + trace.rejected_batch_count, 2);
        assert_eq!(trace.mlp_accumulator_window_count, trace.updates);
        assert_eq!(trace.attention_accumulator_window_count, trace.updates);
        assert_eq!(trace.embedding_accumulator_window_count, trace.updates);
        let line = trace.to_json_line();
        assert!(line.contains("\"batch_windows\":4"));
        assert!(line.contains("\"batch_mode\":\"serial\""));
        assert!(line.contains("\"map_reduce_workers\":1"));
        assert!(line.contains("\"batch_average_shift\":2"));
        assert!(line.contains("\"mlp_accumulator_window_count\""));
        assert!(line.contains("\"attention_accumulator_window_count\""));
        assert!(line.contains("\"embedding_accumulator_window_count\""));
    }

    #[test]
    fn mini_transformer_map_reduce_single_worker_smoke_is_traced() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(4),
                batch_windows: 2,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 17,
                embedding_learning_rate_shift: 13,
                attention_learning_rate_shift: 22,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 16,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::MapReduce,
                map_reduce_workers: 1,
            },
        )
        .expect("map-reduce smoke");

        assert_eq!(trace.examined_windows, 4);
        assert_eq!(trace.accepted_batch_count + trace.rejected_batch_count, 2);
        let line = trace.to_json_line();
        assert!(line.contains("\"batch_mode\":\"map-reduce\""));
        assert!(line.contains("\"map_reduce_workers\":1"));
        assert!(line.contains("\"effective_map_reduce_workers\":1"));
    }

    #[test]
    fn mini_transformer_map_reduce_stacked_accumulates_lower_layers_and_embeddings() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        assert_eq!(initial_model.transformer_layers(), 2);
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(4),
            batch_windows: 2,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 17,
            embedding_learning_rate_shift: 13,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 16,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 1,
        };
        let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
        let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
            tokens,
            &starts,
            config.seq_len,
            config.target_frequency_cap,
            config.target_frequency_min_weight_q15,
        )
        .expect("frequency weights");
        let batch_result = mini_transformer_map_reduce_batch(
            tokens,
            &starts,
            &target_frequency_weights_q15,
            0,
            config.batch_windows,
            0,
            &initial_model,
            config,
            0,
            MiniTransformerTraceDetail::None,
            1,
        )
        .expect("stacked map-reduce batch");

        assert_eq!(batch_result.mlp_weight_gradients.len(), 2);
        assert_eq!(batch_result.attention_weight_gradients.len(), 2);
        assert!(batch_result.embedding_gradient.sample_count > 0);
        assert!(
            batch_result.mlp_weight_gradients[0]
                .down
                .accumulators
                .iter()
                .any(|&value| value != 0)
        );
        assert!(
            batch_result.attention_weight_gradients[0]
                .o
                .accumulators
                .iter()
                .any(|&value| value != 0)
        );

        let run = run_mini_transformer_mlp_training_from_model(tokens, config, initial_model)
            .expect("stacked map-reduce train");
        assert_eq!(run.model.transformer_layers(), 2);
        assert_eq!(run.trace.mlp_accumulator_window_count, run.trace.updates);
        assert_eq!(
            run.trace.attention_accumulator_window_count,
            run.trace.updates
        );
        assert_eq!(
            run.trace.embedding_accumulator_window_count,
            run.trace.updates
        );
    }

    #[test]
    fn mini_transformer_map_reduce_multi_worker_matches_single_worker() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let single_worker = run_mini_transformer_mlp_training_with_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(6),
                batch_windows: 3,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 17,
                embedding_learning_rate_shift: 13,
                attention_learning_rate_shift: 22,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 16,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::MapReduce,
                map_reduce_workers: 1,
            },
        )
        .expect("single-worker map-reduce");
        let multi_worker = run_mini_transformer_mlp_training_with_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                map_reduce_workers: 3,
                ..single_worker.trace.config
            },
        )
        .expect("multi-worker map-reduce");

        assert_eq!(multi_worker.trace.examined_windows, 6);
        assert_eq!(
            multi_worker.trace.accepted_batch_count + multi_worker.trace.rejected_batch_count,
            2
        );
        assert_eq!(multi_worker.model, single_worker.model);
        assert_eq!(
            multi_worker.trace.final_model_hash,
            single_worker.trace.final_model_hash
        );
        assert_eq!(
            multi_worker.trace.output_head_accumulator_window_count,
            single_worker.trace.output_head_accumulator_window_count
        );
        assert_eq!(
            multi_worker.trace.attention_accumulator_window_count,
            single_worker.trace.attention_accumulator_window_count
        );
        let line = multi_worker.trace.to_json_line();
        assert!(line.contains("\"batch_mode\":\"map-reduce\""));
        assert!(line.contains("\"map_reduce_workers\":3"));
        assert!(line.contains("\"effective_map_reduce_workers\":3"));
    }

    #[test]
    fn mini_transformer_map_reduce_matches_serial_with_ascii_lower_adaptive_rule_shifts() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
            .expect("single-layer model");
        let serial = run_mini_transformer_mlp_training_from_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 2,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::AsciiLowerText,
                attention_kind: MiniTransformerAttentionKind::Linear,
                position_policy: MiniTransformerPositionPolicy::Nope,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 17,
                embedding_learning_rate_shift: 13,
                attention_learning_rate_shift: 22,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 16,
                adaptive_rule_shifts: true,
                adaptive_rule_interval_batches: 1,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
            initial_model.clone(),
        )
        .expect("serial adaptive ascii-lower");
        let single_worker = run_mini_transformer_mlp_training_from_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                batch_mode: MiniTransformerBatchMode::MapReduce,
                ..serial.trace.config
            },
            initial_model.clone(),
        )
        .expect("single-worker map-reduce adaptive ascii-lower");
        let multi_worker = run_mini_transformer_mlp_training_from_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                batch_mode: MiniTransformerBatchMode::MapReduce,
                map_reduce_workers: 3,
                ..serial.trace.config
            },
            initial_model,
        )
        .expect("multi-worker map-reduce adaptive ascii-lower");

        assert_eq!(serial.model, single_worker.model);
        assert_eq!(serial.model, multi_worker.model);
        assert_eq!(
            serial.trace.final_model_hash,
            single_worker.trace.final_model_hash
        );
        assert_eq!(
            serial.trace.final_model_hash,
            multi_worker.trace.final_model_hash
        );
        assert!(multi_worker.trace.adaptive_rule_update_count > 0);
        let line = multi_worker.trace.to_json_line();
        assert!(line.contains("\"tokenizer\":\"byte_ascii_lower_text_u8_v1\""));
        assert!(line.contains("\"adaptive_rule_shifts\":true"));
        assert!(line.contains("\"batch_mode\":\"map-reduce\""));
    }

    #[test]
    fn linear_weight_gradient_i64_averages_then_updates_i8() {
        let mut gradient = LinearWeightGradientI64::new(2, 2).expect("gradient");
        let input = [4096_i16, 8192_i16];
        let scaled_grad_output = [1024_i32, 2048_i32];

        accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
            .expect("first sample");
        accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
            .expect("second sample");

        let mut weights = [10_i8, 10_i8, 10_i8, 10_i8];
        let stats =
            apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 22, false)
                .expect("apply");

        assert_eq!(weights, [9, 8, 8, 6]);
        assert_eq!(stats.gradient_saturation_count, 0);
        assert_eq!(stats.zero_delta_count, 0);
        assert_eq!(stats.weight_delta_l1, 9);
        assert_eq!(gradient.sample_count, 0);
        assert!(gradient.accumulators.iter().all(|&value| value == 0));
    }

    #[test]
    fn linear_weight_gradient_i64_carries_subthreshold_residuals() {
        let mut gradient = LinearWeightGradientI64::new(1, 1).expect("gradient");
        let input = [1_i16];
        let scaled_grad_output = [1_i32];
        let mut weights = [10_i8];

        accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
            .expect("first sample");
        let first = apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 2, true)
            .expect("first apply");

        assert_eq!(weights, [10]);
        assert_eq!(first.zero_delta_count, 1);
        assert_eq!(first.weight_delta_l1, 0);
        assert_eq!(gradient.residuals, [1]);

        accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
            .expect("second sample");
        let second =
            apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 2, true)
                .expect("second apply");

        assert_eq!(weights, [9]);
        assert_eq!(second.zero_delta_count, 0);
        assert_eq!(second.weight_delta_l1, 1);
        assert_eq!(gradient.residuals, [-2]);
    }

    #[test]
    fn mini_transformer_effective_batch_shift_compensates_batch_average() {
        assert_eq!(
            mini_transformer_component_shift_for_effective_batch_shift(18, 8).expect("shift"),
            15
        );
        assert_eq!(
            mini_transformer_component_shift_for_effective_batch_shift(3, 8).expect("shift"),
            0
        );
        assert!(mini_transformer_component_shift_for_effective_batch_shift(2, 8).is_err());
    }

    #[test]
    fn linear_weight_gradient_i64_uses_effective_batch_shift() {
        let mut gradient = LinearWeightGradientI64::new(1, 1).expect("gradient");
        let input = [1_i16];
        let scaled_grad_output = [1_i32];
        for _ in 0..8 {
            accumulate_linear_weight_gradient_i64_prescaled(
                &input,
                &scaled_grad_output,
                &mut gradient,
            )
            .expect("sample");
        }
        let mut compensated_weights = [10_i8];
        let compensated_shift =
            mini_transformer_component_shift_for_effective_batch_shift(3, 8).expect("shift");
        let compensated = apply_linear_weight_gradient_i64_to_i8(
            &mut gradient,
            &mut compensated_weights,
            1,
            compensated_shift,
            true,
        )
        .expect("compensated apply");

        assert_eq!(compensated_weights, [9]);
        assert_eq!(compensated.zero_delta_count, 0);
        assert_eq!(compensated.weight_delta_l1, 1);

        let mut uncompensated = LinearWeightGradientI64::new(1, 1).expect("gradient");
        for _ in 0..8 {
            accumulate_linear_weight_gradient_i64_prescaled(
                &input,
                &scaled_grad_output,
                &mut uncompensated,
            )
            .expect("sample");
        }
        let mut uncompensated_weights = [10_i8];
        let uncompensated_stats = apply_linear_weight_gradient_i64_to_i8(
            &mut uncompensated,
            &mut uncompensated_weights,
            1,
            3,
            true,
        )
        .expect("uncompensated apply");

        assert_eq!(uncompensated_weights, [10]);
        assert_eq!(uncompensated_stats.zero_delta_count, 1);
        assert_eq!(uncompensated_stats.weight_delta_l1, 0);
        assert_eq!(uncompensated.residual_l1(), 1);
    }

    #[test]
    fn gated_mlp_weight_gradient_i64_averages_then_updates_i8() {
        let scales = [FixedScale {
            multiplier: 1,
            right_shift: 0,
        }; 2];
        let input = [4096_i16, 8192_i16];
        let grad_output = [1024_i16, -1024_i16];
        let forward_gated = [4096_i16, -4096_i16];
        let grad_up = [2048_i16, -1024_i16];
        let grad_gate = [-2048_i16, 1024_i16];
        let params = GatedMlpWeightUpdateParams {
            up_scales: &scales,
            gate_scales: &scales,
            down_scales: &scales,
            down_to_hidden_scales: &scales,
            seq_len: 1,
            d_model: 2,
            hidden_dim: 2,
            learning_rate: 1,
            learning_rate_shift: 22,
        };
        let mut gradient = GatedMlpWeightGradientI64::new(2, 2).expect("gradient");
        let mut scaled = [0_i32; 2];

        accumulate_gated_mlp_weight_gradient_i64(
            &input,
            &grad_output,
            &forward_gated,
            &grad_up,
            &grad_gate,
            params,
            &mut gradient,
            &mut scaled,
        )
        .expect("first sample");
        accumulate_gated_mlp_weight_gradient_i64(
            &input,
            &grad_output,
            &forward_gated,
            &grad_up,
            &grad_gate,
            params,
            &mut gradient,
            &mut scaled,
        )
        .expect("second sample");

        let mut up_weights = [10_i8; 4];
        let mut gate_weights = [10_i8; 4];
        let mut down_weights = [10_i8; 4];
        let stats = apply_gated_mlp_weight_gradient_i64_to_i8(
            &mut gradient,
            &mut up_weights,
            &mut gate_weights,
            &mut down_weights,
            1,
            22,
            false,
        )
        .expect("apply");

        assert_eq!(down_weights, [9, 11, 11, 9]);
        assert_eq!(up_weights, [8, 6, 11, 12]);
        assert_eq!(gate_weights, [12, 14, 9, 8]);
        assert_eq!(stats.gradient_saturation_count(), Some(0));
        assert_eq!(stats.zero_delta_count(), Some(0));
        assert_eq!(stats.weight_delta_l1(), Some(22));
        assert_eq!(gradient.down.sample_count, 0);
        assert_eq!(gradient.up.sample_count, 0);
        assert_eq!(gradient.gate.sample_count, 0);
    }

    #[cfg(not(feature = "mini-calibrated"))]
    #[test]
    fn attention_weight_gradient_i64_averages_then_updates_i8() {
        let mut embedding_output = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        embedding_output[0] = 4096;
        embedding_output[1] = 8192;
        let mut attention_context = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        attention_context[0] = 4096;
        attention_context[1] = -4096;
        let cache = MiniTransformerBlockForwardCache {
            block_input: embedding_output.clone(),
            attention_norm: embedding_output.clone(),
            attention_q: Vec::new(),
            attention_k: Vec::new(),
            attention_v: Vec::new(),
            attention_context,
            attention_probabilities_q15: Vec::new(),
            attention_output: Vec::new(),
            attention_residual: Vec::new(),
            mlp_norm: Vec::new(),
            mlp_up: Vec::new(),
            mlp_gate: Vec::new(),
            mlp_gated: Vec::new(),
            mlp_output: Vec::new(),
            block_output: Vec::new(),
            residual_saturation_count: 0,
        };
        let mut grad_q = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_k = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_v = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_o = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        grad_q[0] = 1024;
        grad_k[1] = 1024;
        grad_v[0] = -1024;
        grad_o[0] = 1024;
        grad_o[1] = -1024;
        let mut gradient = MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL)
            .expect("gradient");
        let mut scaled = [0_i32; MINI_TRANSFORMER_D_MODEL];

        accumulate_mini_transformer_attention_weight_gradient_i64(
            &cache,
            &grad_o,
            &grad_q,
            &grad_k,
            &grad_v,
            &mut gradient,
            &mut scaled,
        )
        .expect("first sample");
        accumulate_mini_transformer_attention_weight_gradient_i64(
            &cache,
            &grad_o,
            &grad_q,
            &grad_k,
            &grad_v,
            &mut gradient,
            &mut scaled,
        )
        .expect("second sample");

        let mut model = MiniTransformerMlpModel {
            context_seq_len: 1,
            embeddings: vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
            position_embeddings: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
            attention_rms_weights: Vec::new(),
            mlp_rms_weights: Vec::new(),
            q_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            k_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            v_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            o_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            up_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
            gate_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
            down_weights: vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
            output_weights: vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
        };
        let stats = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
            &mut gradient,
            &mut model,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: Some(1),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 12,
                attention_learning_rate_shift: 22,
                attention_q_learning_rate_shift: 22,
                attention_qk_learning_rate_shift: 22,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("apply");

        let row_1 = MINI_TRANSFORMER_D_MODEL;
        assert_eq!(&model.q_weights[..4], &[9, 8, 10, 10]);
        assert_eq!(&model.k_weights[row_1..row_1 + 4], &[9, 8, 10, 10]);
        assert_eq!(&model.v_weights[..4], &[11, 12, 10, 10]);
        assert_eq!(&model.o_weights[..4], &[9, 11, 10, 10]);
        assert_eq!(&model.o_weights[row_1..row_1 + 4], &[11, 9, 10, 10]);
        assert_eq!(stats.gradient_saturation_count, 0);
        assert_eq!(stats.zero_delta_count, 0);
        assert_eq!(stats.weight_delta_l1, 13);
        assert_eq!(gradient.q.sample_count, 0);
        assert_eq!(gradient.k.sample_count, 0);
        assert_eq!(gradient.v.sample_count, 0);
        assert_eq!(gradient.o.sample_count, 0);
    }

    #[test]
    fn attention_qk_gradient_i64_carries_subthreshold_residuals() {
        let mut gradient = MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL)
            .expect("gradient");
        let mut model = MiniTransformerMlpModel {
            context_seq_len: 1,
            embeddings: vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
            position_embeddings: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
            attention_rms_weights: Vec::new(),
            mlp_rms_weights: Vec::new(),
            q_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            k_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            v_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            o_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
            up_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
            gate_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
            down_weights: vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
            output_weights: vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
        };
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: Some(1),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 12,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 2,
            attention_qk_learning_rate_shift: 2,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        };

        gradient.q.accumulators[0] = 1;
        gradient.q.sample_count = 1;
        gradient.k.accumulators[0] = 1;
        gradient.k.sample_count = 1;
        let first = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
            &mut gradient,
            &mut model,
            config,
        )
        .expect("first apply");

        assert_eq!(model.q_weights[0], 10);
        assert_eq!(model.k_weights[0], 10);
        assert_eq!(first.weight_delta_l1, 0);
        assert_eq!(first.zero_delta_count, 2);
        assert_eq!(gradient.q.residuals[0], 1);
        assert_eq!(gradient.k.residuals[0], 1);

        gradient.q.accumulators[0] = 1;
        gradient.q.sample_count = 1;
        gradient.k.accumulators[0] = 1;
        gradient.k.sample_count = 1;
        let second = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
            &mut gradient,
            &mut model,
            config,
        )
        .expect("second apply");

        assert_eq!(model.q_weights[0], 9);
        assert_eq!(model.k_weights[0], 9);
        assert_eq!(second.weight_delta_l1, 2);
        assert_eq!(second.zero_delta_count, 0);
    }

    #[test]
    fn embedding_gradient_i64_averages_then_updates_i16() {
        let context = [1_u8, 2_u8];
        let mut grad_embedding_output = vec![0_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
        grad_embedding_output[..4].copy_from_slice(&[4096, -4096, 0, 8192]);
        grad_embedding_output[MINI_TRANSFORMER_D_MODEL..MINI_TRANSFORMER_D_MODEL + 4]
            .copy_from_slice(&[-4096, 0, 4096, 0]);
        let mut gradient =
            MiniTransformerEmbeddingGradientI64::new(context.len()).expect("gradient");

        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
            &context,
            &grad_embedding_output,
            MiniTransformerPositionPolicy::LearnedAbsolute,
            &mut gradient,
        )
        .expect("first sample");
        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
            &context,
            &grad_embedding_output,
            MiniTransformerPositionPolicy::LearnedAbsolute,
            &mut gradient,
        )
        .expect("second sample");

        let mut embeddings = vec![10_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let mut position_embeddings = vec![10_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
        let stats = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
            &mut gradient,
            &mut embeddings,
            &mut position_embeddings,
            MiniTransformerPositionPolicy::LearnedAbsolute,
            1,
            12,
        )
        .expect("apply");

        let row_1 = usize::from(context[0]) * MINI_TRANSFORMER_D_MODEL;
        let row_2 = usize::from(context[1]) * MINI_TRANSFORMER_D_MODEL;
        let position_row_1 = 0;
        let position_row_2 = MINI_TRANSFORMER_D_MODEL;
        assert_eq!(&embeddings[row_1..row_1 + 4], &[9, 11, 10, 8]);
        assert_eq!(&embeddings[row_2..row_2 + 4], &[11, 10, 9, 10]);
        assert_eq!(
            &position_embeddings[position_row_1..position_row_1 + 4],
            &[9, 11, 10, 8]
        );
        assert_eq!(
            &position_embeddings[position_row_2..position_row_2 + 4],
            &[11, 10, 9, 10]
        );
        assert_eq!(stats.gradient_saturation_count, 0);
        assert_eq!(stats.zero_delta_count, 0);
        assert_eq!(stats.weight_delta_l1, 12);
        assert_eq!(gradient.sample_count, 0);
        assert!(gradient.token_accumulators.iter().all(|&value| value == 0));
        assert!(
            gradient
                .position_accumulators
                .iter()
                .all(|&value| value == 0)
        );
    }

    #[test]
    fn embedding_gradient_i64_carries_subthreshold_residuals() {
        let mut gradient = MiniTransformerEmbeddingGradientI64::new(1).expect("gradient");
        let token = 7_u8;
        let context = [token];
        let mut grad_embedding_output = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        grad_embedding_output[0] = 1;
        let row = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        let mut embeddings = vec![10_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let mut position_embeddings = Vec::new();

        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
            &context,
            &grad_embedding_output,
            MiniTransformerPositionPolicy::Nope,
            &mut gradient,
        )
        .expect("first sample");
        let first = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
            &mut gradient,
            &mut embeddings,
            &mut position_embeddings,
            MiniTransformerPositionPolicy::Nope,
            1,
            2,
        )
        .expect("first apply");

        assert_eq!(embeddings[row], 10);
        assert_eq!(first.zero_delta_count, 1);
        assert_eq!(first.weight_delta_l1, 0);
        assert_eq!(gradient.residual_l1(MiniTransformerPositionPolicy::Nope), 1);

        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
            &context,
            &grad_embedding_output,
            MiniTransformerPositionPolicy::Nope,
            &mut gradient,
        )
        .expect("second sample");
        let second = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
            &mut gradient,
            &mut embeddings,
            &mut position_embeddings,
            MiniTransformerPositionPolicy::Nope,
            1,
            2,
        )
        .expect("second apply");

        assert_eq!(embeddings[row], 9);
        assert_eq!(second.zero_delta_count, 0);
        assert_eq!(second.weight_delta_l1, 1);
    }

    #[test]
    fn byte_window_starts_remain_sequential() {
        let starts = byte_window_starts(1000, 4, 10, 0, Some(5));
        assert_eq!(starts, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn mini_transformer_window_starts_spread_capped_runs() {
        let starts = mini_transformer_window_starts(1000, 4, 10, 0, Some(5));
        assert_eq!(starts, vec![0, 250, 500, 740, 990]);
    }

    #[test]
    fn mini_transformer_window_starts_keep_full_runs_sequential() {
        let sequential = byte_window_starts(1000, 4, 10, 0, None);
        let distributed = mini_transformer_window_starts(1000, 4, 10, 0, None);
        assert_eq!(distributed, sequential);
    }

    #[test]
    fn mini_transformer_filtered_window_starts_cap_after_target_filter() {
        let mut tokens = vec![b'a'; 40];
        for target_index in [4_usize, 10, 16, 22, 28, 34] {
            tokens[target_index] = b'Z';
        }

        let starts = mini_transformer_filtered_window_starts(
            tokens.len(),
            &tokens,
            MiniTransformerMlpTrainConfig {
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(3),
                target_token_min: b'Z',
                target_token_max: b'Z',
                ..MiniTransformerMlpTrainConfig::default()
            },
        );

        assert_eq!(starts, vec![0, 18, 30]);
        assert!(starts.iter().all(|&start| tokens[start + 4] == b'Z'));
    }

    #[test]
    fn mini_transformer_filtered_window_starts_can_target_marker_segment() {
        let tokens = [
            0, 1, 2, b's', b'e', 3, b'A', b'B', 5, 1, 2, b'x', 3, b'C', 4, b'i', 5,
        ];
        let starts = mini_transformer_filtered_window_starts(
            tokens.len(),
            &tokens,
            MiniTransformerMlpTrainConfig {
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: None,
                target_token_min: b'A',
                target_token_max: b'Z',
                target_segment: MiniTransformerTargetSegment::after_marker_before_any(3, &[4, 5])
                    .expect("segment"),
                ..MiniTransformerMlpTrainConfig::default()
            },
        );

        assert_eq!(starts, vec![5, 6, 12]);
        assert_eq!(
            starts
                .iter()
                .map(|&start| tokens[start + 1])
                .collect::<Vec<_>>(),
            vec![b'A', b'B', b'C']
        );
    }

    #[test]
    fn mini_transformer_filtered_window_starts_can_target_sequence_segment() {
        let tokens = [
            1, 2, b'p', 3, b'S', b'o', b'B', b'a', b':', b' ', b'H', 5, 1, 2, b'q', 3, b'S', b'o',
            b'C', b'a', b'm', b':', 5,
        ];
        let starts = mini_transformer_filtered_window_starts(
            tokens.len(),
            &tokens,
            MiniTransformerMlpTrainConfig {
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: None,
                target_token_min: b'A',
                target_token_max: b'z',
                target_segment: MiniTransformerTargetSegment::after_sequence_before_any(
                    &[3, b'S', b'o'],
                    &[b':', 4, 5],
                )
                .expect("segment"),
                ..MiniTransformerMlpTrainConfig::default()
            },
        );

        assert_eq!(starts, vec![5, 6, 17, 18, 19]);
        assert_eq!(
            starts
                .iter()
                .map(|&start| tokens[start + 1])
                .collect::<Vec<_>>(),
            vec![b'B', b'a', b'C', b'a', b'm']
        );
    }

    #[test]
    fn mini_transformer_filtered_window_starts_can_target_first_after_sequence() {
        let tokens = [
            1, 3, b'H', b'e', b' ', b'm', b'a', 5, 1, 3, b'H', b'e', b' ', b'i', b's', 5,
        ];
        let starts = mini_transformer_filtered_window_starts(
            tokens.len(),
            &tokens,
            MiniTransformerMlpTrainConfig {
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: None,
                target_token_min: b'a',
                target_token_max: b'z',
                target_segment: MiniTransformerTargetSegment::first_after_sequence_before_any(
                    b"He ",
                    &[4, 5],
                )
                .expect("segment"),
                ..MiniTransformerMlpTrainConfig::default()
            },
        );

        assert_eq!(starts, vec![4, 12]);
        assert_eq!(
            starts
                .iter()
                .map(|&start| tokens[start + 1])
                .collect::<Vec<_>>(),
            vec![b'm', b'i']
        );
    }

    #[test]
    fn mini_transformer_loss_guard_starts_mix_batch_and_global_points() {
        let starts: Vec<usize> = (0..32).map(|index| index * 10).collect();
        let guarded = mini_transformer_loss_guard_starts(&starts, 5, 7);

        assert!(guarded.contains(&50));
        assert!(guarded.contains(&60));
        assert_eq!(guarded.first().copied(), Some(50));
        assert_eq!(guarded.get(1).copied(), Some(60));
        assert!(guarded.contains(&0));
        assert!(guarded.contains(&310));
        assert_eq!(guarded.len(), 17);
    }

    #[test]
    fn mini_transformer_loss_guard_ignores_small_regressions() {
        assert!(!mini_transformer_loss_guard_regressed(100_000, 117_000, 17));
        assert!(mini_transformer_loss_guard_regressed(100_000, 118_000, 17));
    }

    #[test]
    fn attention_vo_oracle_does_not_increase_configured_loss() {
        let tokens = b"To be or not to be, that is the question. To be or not to be. ";
        let seq_len = 4;
        let starts = byte_window_starts(tokens.len(), seq_len, 1, 0, Some(4));
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(seq_len);
        if MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL {
            assert_eq!(
                mini_transformer_attention_vo_oracle_update_i8_checked(
                    &mut model, tokens, &starts, seq_len, 1,
                ),
                Err(TrainError::InvalidConfig)
            );
            return;
        }
        let before = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
            .expect("before loss");
        let (v, o) = mini_transformer_attention_vo_oracle_update_i8_checked(
            &mut model, tokens, &starts, seq_len, 1,
        )
        .expect("oracle update");
        let after = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
            .expect("after loss");

        assert!(after <= before);
        assert_eq!(v.gradient_saturation_count, 0);
        assert_eq!(o.gradient_saturation_count, 0);
        assert_eq!(
            v.zero_delta_count + v.weight_delta_l1 as usize,
            MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
        );
        assert_eq!(
            o.zero_delta_count + o.weight_delta_l1 as usize,
            MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
        );
    }

    #[test]
    fn mini_transformer_model_round_trips_and_generates() {
        let tokens = b"To be or not to be, that is the question. To be or not to be. ";
        let run = run_mini_transformer_mlp_training_with_model(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(32),
                batch_windows: 1,
                target_token_min: u8::MIN,
                target_token_max: u8::MAX,
                target_segment: MiniTransformerTargetSegment::All,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                argmax_margin_weight_q15: 0,
                tokenizer_id: ByteTokenizerId::Identity,
                attention_kind: MiniTransformerAttentionKind::Base2Softmax,
                position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 18,
                adaptive_rule_shifts: false,
                adaptive_rule_interval_batches:
                    DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
                adaptive_attention_shifts: false,
                adaptive_holographic_shifts: false,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
                batch_mode: MiniTransformerBatchMode::Serial,
                map_reduce_workers: 1,
            },
        )
        .expect("mini train");
        let bytes = run.model.to_bytes();
        let decoded = MiniTransformerMlpModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.model_hash(), run.trace.final_model_hash);
        assert_eq!(decoded.embedding_hash(), run.trace.final_embedding_hash);
        assert_eq!(decoded.attention_hash(), run.trace.final_attention_hash);
        assert_eq!(decoded.mlp_hash(), run.trace.final_mlp_hash);
        assert_eq!(decoded.output_head_hash(), run.trace.final_output_head_hash);

        let generation =
            generate_mini_transformer(&decoded, b"To be", ByteGenerationConfig::greedy(8))
                .expect("generate");

        assert_eq!(generation.generated_bytes.len(), 8);
        assert_eq!(generation.steps.len(), 8);
        assert_eq!(
            generation.attention_kind,
            MiniTransformerAttentionKind::Base2Softmax
        );
        assert_eq!(generation.context_seq_len, decoded.context_seq_len);
        assert_eq!(generation.model_hash, decoded.model_hash());
        assert_eq!(generation.attention_hash, decoded.attention_hash());
        assert_eq!(generation.mlp_hash, decoded.mlp_hash());
        assert_eq!(generation.output_head_hash, decoded.output_head_hash());
        let line = generation.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.mini_transformer_generation_trace.v1\""));
        assert!(line.contains("\"model\":\"mini_transformer_byte_qkvo_mlp_v1\""));
        assert!(line.contains("\"attention_kind\":\"base2_softmax\""));
    }

    #[test]
    fn mini_transformer_generation_can_use_linear_attention() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(16);
        let generation = generate_mini_transformer_with_attention_kind(
            &model,
            b"To be",
            ByteGenerationConfig::greedy(4),
            MiniTransformerAttentionKind::Linear,
        )
        .expect("linear generation");

        assert_eq!(generation.generated_bytes.len(), 4);
        assert_eq!(generation.steps.len(), 4);
        assert_eq!(generation.context_seq_len, 16);
        assert_eq!(
            generation.attention_kind,
            MiniTransformerAttentionKind::Linear
        );
        let line = generation.to_json_line();
        assert!(line.contains("\"attention_kind\":\"linear\""));
    }

    #[test]
    fn mini_transformer_generation_can_use_streaming_linear_attention_nope() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(16, 1)
            .expect("single-layer model");
        let generation = generate_mini_transformer_with_attention_kind(
            &model,
            b"To be",
            ByteGenerationConfig::greedy(4),
            MiniTransformerAttentionKind::LinearStreamingNope,
        )
        .expect("streaming linear generation");

        assert_eq!(generation.generated_bytes.len(), 4);
        assert_eq!(generation.steps.len(), 4);
        assert_eq!(generation.context_seq_len, 16);
        assert_eq!(
            generation.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingNope
        );
        let line = generation.to_json_line();
        assert!(line.contains("\"attention_kind\":\"linear_streaming_nope\""));
        assert!(line.contains("\"position_policy\":\"nope\""));
        assert!(line.contains("\"incremental_attention_state\":true"));
        assert!(line.contains("\"streaming_nope_ignores_learned_position_embeddings\""));
        assert!(!line.contains("\"no_kv_cache_yet\""));
    }

    #[test]
    fn mini_transformer_generation_can_use_streaming_linear_ttt_attention_nope() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(16, 1)
            .expect("single-layer model");
        let generation =
            generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
                &model,
                b"To be",
                ByteGenerationConfig::greedy(4),
                MiniTransformerAttentionKind::LinearStreamingTttNope,
                MiniTransformerPositionPolicy::Nope,
                None,
                DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT,
            )
            .expect("streaming linear ttt generation");

        assert_eq!(generation.generated_bytes.len(), 4);
        assert_eq!(generation.steps.len(), 4);
        assert_eq!(
            generation.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingTttNope
        );
        let stats = generation.ttt_stats.expect("ttt stats");
        assert_eq!(
            stats.learning_rate_shift,
            DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT
        );
        assert_eq!(stats.step_count, b"To be".len() + 4);
        assert!(stats.total_state_delta_l1 > 0);
        let line = generation.to_json_line();
        assert!(line.contains("\"attention_kind\":\"linear_streaming_ttt_nope\""));
        assert!(line.contains("\"incremental_attention_state\":true"));
        assert!(line.contains("\"ttt\":{\"learning_rate_shift\":"));
    }

    #[test]
    fn mini_transformer_generation_left_pads_short_prompt() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(16);
        let short = generate_mini_transformer(&model, b"a", ByteGenerationConfig::greedy(1))
            .expect("short");
        let mut padded_prompt = vec![b' '; 15];
        padded_prompt.push(b'a');
        let explicit =
            generate_mini_transformer(&model, &padded_prompt, ByteGenerationConfig::greedy(1))
                .expect("explicit");

        assert_eq!(short.context_seq_len, 16);
        assert_eq!(short.steps[0], explicit.steps[0]);
        assert_eq!(short.generated_bytes, explicit.generated_bytes);
    }

    #[test]
    fn mini_transformer_eval_is_deterministic_and_read_only() {
        let tokens = b"crowley shakespeare blake literary evaluation fixture";
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
        let config = MiniTransformerMlpEvalConfig {
            seq_len: 8,
            stride: 3,
            max_windows: Some(7),
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
        };
        let before_hash = model.model_hash();
        let left = evaluate_mini_transformer_mlp_model(tokens, &model, config).expect("left");
        let right = evaluate_mini_transformer_mlp_model(tokens, &model, config).expect("right");

        assert_eq!(left, right);
        assert_eq!(left.windows, 7);
        assert_eq!(left.model_hash, before_hash);
        assert_eq!(model.model_hash(), before_hash);
        assert_eq!(left.invalid_forward_count, 0);
        assert!(left.unique_predicted_tokens > 0);
        assert!(left.unique_predicted_tokens <= BYTE_VOCAB);
        assert!(left.most_predicted_token.is_some());
        assert!(left.most_predicted_token_count <= left.windows);
        assert_eq!(
            left.most_predicted_token_share_per_mille,
            left.most_predicted_token_count * 1000 / left.windows
        );
        let json = left.to_json_line();
        assert!(json.contains(MINI_TRANSFORMER_EVAL_SCHEMA));
        assert!(json.contains("\"most_predicted_token_share_per_mille\":"));
    }

    #[test]
    fn mini_transformer_block_expert_zero_identity_and_artifact_are_locked() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let expert =
            MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 17).expect("block expert");
        let base = mini_transformer_next_token_row_with_attention_kind_position_policy(
            &model,
            b"Blak",
            MiniTransformerAttentionKind::Base2Softmax,
            MiniTransformerPositionPolicy::LearnedAbsolute,
        )
        .expect("base row");
        let adapted = mini_transformer_next_token_row_with_block_expert(
            &model,
            &expert,
            b"Blak",
            MiniTransformerAttentionKind::Base2Softmax,
            MiniTransformerPositionPolicy::LearnedAbsolute,
        )
        .expect("adapted row");
        assert_eq!(adapted, base);

        let bytes = expert.to_bytes();
        assert_eq!(
            MiniTransformerBlockLowRankExpert::from_bytes(&bytes).expect("decode"),
            expert
        );
        let mut corrupt = bytes;
        corrupt[64] ^= 1;
        assert!(MiniTransformerBlockLowRankExpert::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn mini_transformer_block_expert_training_updates_only_expert() {
        let tokens = b"Crowley Shakespeare Blake sing through the integer swarm.";
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let model_hash = model.model_hash();
        let mut expert =
            MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 23).expect("block expert");
        let stats = train_mini_transformer_block_expert(
            tokens,
            &model,
            &mut expert,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                max_windows: Some(4),
                batch_windows: 2,
                ..MiniTransformerMlpTrainConfig::default()
            },
            2,
            1024,
            0,
        )
        .expect("train block expert");
        assert_eq!(model.model_hash(), model_hash);
        assert_eq!(stats.optimizer_steps, 2);
        assert!(stats.weight_delta_l1 > 0);
        assert!(
            expert
                .expansion_weights_q15
                .iter()
                .any(|&weight| weight != 0)
        );
    }

    #[test]
    fn mini_transformer_block_expert_raw_probability_gradient_and_guard_are_locked() {
        let tokens = b"Crowley Shakespeare Blake sing through the integer swarm.";
        let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 2)
            .expect("two-layer model");
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            max_windows: Some(4),
            batch_windows: 4,
            ..MiniTransformerMlpTrainConfig::default()
        };

        let mut unguarded =
            MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 29).expect("expert");
        let unguarded_stats = train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
            tokens,
            &model,
            &mut unguarded,
            config,
            4,
            1024,
            0,
            Some(1),
            false,
            MiniTransformerBlockExpertObjective::ProbabilityError,
        )
        .expect("metric-aligned update");
        assert_eq!(unguarded_stats.optimizer_steps, 1);
        assert!(unguarded_stats.weight_delta_l1 > 0);
        let parameters_per_layer = MINI_TRANSFORMER_D_MODEL * unguarded.rank;
        assert!(
            unguarded.expansion_weights_q15[..parameters_per_layer]
                .iter()
                .all(|&weight| weight == 0)
        );
        assert!(
            unguarded.expansion_weights_q15[parameters_per_layer..]
                .iter()
                .any(|&weight| weight != 0)
        );

        let mut guarded =
            MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 29).expect("expert");
        let baseline = evaluate_mini_transformer_block_expert(
            tokens,
            &model,
            &guarded,
            MiniTransformerMlpEvalConfig {
                seq_len: 4,
                stride: 1,
                max_windows: Some(4),
                attention_kind: config.attention_kind,
                position_policy: config.position_policy,
            },
        )
        .expect("baseline");
        let guarded_stats = train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
            tokens,
            &model,
            &mut guarded,
            config,
            4,
            1024,
            0,
            Some(1),
            true,
            MiniTransformerBlockExpertObjective::ProbabilityError,
        )
        .expect("guarded update");
        let final_metrics = evaluate_mini_transformer_block_expert(
            tokens,
            &model,
            &guarded,
            MiniTransformerMlpEvalConfig {
                seq_len: 4,
                stride: 1,
                max_windows: Some(4),
                attention_kind: config.attention_kind,
                position_policy: config.position_policy,
            },
        )
        .expect("final metrics");
        assert!(final_metrics.probability_error_q15 <= baseline.probability_error_q15);
        assert_eq!(
            guarded_stats.accepted_forward_steps
                + guarded_stats.accepted_reverse_steps
                + guarded_stats.rejected_steps,
            guarded_stats.optimizer_steps
        );
        assert!(
            guarded.expansion_weights_q15[..parameters_per_layer]
                .iter()
                .all(|&weight| weight == 0)
        );
    }
}
