use nsrl_core::{
    FixedScale, GatedMlpBackwardScales, GatedMlpBackwardWorkspace, GatedMlpI16Params,
    GatedMlpWeightUpdateParams, GatedMlpWeightUpdateStats, GatedMlpWeightUpdateWorkspace,
    GatedMlpWorkspace, LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace,
    LinearBackwardWeightUpdateI8Params, LinearBackwardWeightUpdateWorkspace, LinearI16I8Params,
    LinearWeightUpdateStats, MASKED_LOGIT, MAX_RIGHT_SHIFT, Q15_SHIFT, SelfAttentionI16Params,
    SelfAttentionWorkspace, attention_dot_q_k_i16_i32_checked, base2_softmax_i32_q15,
    gated_mlp_backward_input_i16_q15_checked, gated_mlp_backward_weight_update_i8_checked,
    gated_mlp_i16_q15_checked, linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_backward_weight_update_i8_checked, linear_i16_i8_i16_per_channel_checked,
    round_shift_rhu_i64, saturate_i8, saturate_i16, self_attention_i16_q15_checked,
    sqrt_power_of_four_shift,
};

pub const SCHEMA: &str = "nsrl.training_smoke_trace.v1";
pub const SOFTMAX_SCHEMA: &str = "nsrl.training_softmax_trace.v1";
pub const LINEAR_BACKWARD_SCHEMA: &str = "nsrl.training_linear_backward_trace.v1";
pub const GATED_MLP_BACKWARD_SCHEMA: &str = "nsrl.training_gated_mlp_backward_trace.v1";
pub const BYTE_SOFTMAX_SCHEMA: &str = "nsrl.training_byte_softmax_trace.v1";
pub const BYTE_GENERATION_SCHEMA: &str = "nsrl.byte_generation_trace.v1";
pub const BYTE_EMBED_SOFTMAX_SCHEMA: &str = "nsrl.training_byte_embed_softmax_trace.v1";
pub const BYTE_EMBED_GENERATION_SCHEMA: &str = "nsrl.byte_embed_generation_trace.v1";
pub const MINI_TRANSFORMER_MLP_SCHEMA: &str = "nsrl.training_mini_transformer_mlp_trace.v1";
pub const MINI_TRANSFORMER_GENERATION_SCHEMA: &str = "nsrl.mini_transformer_generation_trace.v1";
pub const AUTHORITY: &str = "deterministic_training_replay";
pub const GENERATION_AUTHORITY: &str = "deterministic_integer_generation";
pub const TASK: &str = "tiny_next_char_output_head";
pub const SOFTMAX_TASK: &str = "tiny_next_char_output_head_base2_softmax";
pub const LINEAR_BACKWARD_TASK: &str = "tiny_linear_layer_backward";
pub const GATED_MLP_BACKWARD_TASK: &str = "tiny_gated_mlp_weight_backward";
pub const BYTE_SOFTMAX_TASK: &str = "wiki_bard_byte_next_token_output_head";
pub const BYTE_EMBED_SOFTMAX_TASK: &str = "wiki_bard_byte_next_token_embedding_output_head";
pub const MINI_TRANSFORMER_MLP_TASK: &str = "wiki_bard_mini_transformer_mlp_first";
pub const BYTE_TOKENIZER_ID: &str = "byte_identity_u8_v1";
pub const BYTE_SOFTMAX_MODEL_ID: &str = "byte_softmax_bigram_output_head_v1";
pub const BYTE_SOFTMAX_MODEL_MAGIC: &[u8; 8] = b"NSRLBM1\n";
pub const BYTE_EMBED_SOFTMAX_MODEL_ID: &str = "byte_embed_softmax_context_head_v1";
pub const BYTE_EMBED_SOFTMAX_MODEL_MAGIC: &[u8; 8] = b"NSRLEM1\n";
pub const MINI_TRANSFORMER_MODEL_ID: &str = "mini_transformer_byte_qkvo_mlp_v1";
pub const MINI_TRANSFORMER_MODEL_MAGIC: &[u8; 8] = b"NSRLMT1\n";
pub const VOCAB: usize = 4;
pub const D_MODEL: usize = 8;
pub const BYTE_VOCAB: usize = 256;
pub const BYTE_D_MODEL: usize = 257;
pub const BYTE_EMBED_DIM: usize = 32;
pub const BYTE_EMBED_D_MODEL: usize = BYTE_EMBED_DIM + 1;
pub const LINEAR_BACKWARD_INPUT_DIM: usize = 4;
pub const LINEAR_BACKWARD_OUTPUT_DIM: usize = 3;
pub const GATED_MLP_BACKWARD_SEQ_LEN: usize = 1;
pub const GATED_MLP_BACKWARD_D_MODEL: usize = 2;
pub const GATED_MLP_BACKWARD_HIDDEN_DIM: usize = 4;
const GATED_MLP_BACKWARD_SCALED_WORKSPACE_DIM: usize =
    if GATED_MLP_BACKWARD_D_MODEL > GATED_MLP_BACKWARD_HIDDEN_DIM {
        GATED_MLP_BACKWARD_D_MODEL
    } else {
        GATED_MLP_BACKWARD_HIDDEN_DIM
    };
pub const MINI_TRANSFORMER_D_MODEL: usize = 4;
pub const MINI_TRANSFORMER_HEADS: usize = 1;
pub const MINI_TRANSFORMER_HIDDEN_DIM: usize = 8;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_EPOCHS: usize = 8;
const DEFAULT_LEARNING_RATE: i8 = 4;
const DEFAULT_SOFTMAX_LEARNING_RATE: i32 = 1;
const DEFAULT_SOFTMAX_LEARNING_RATE_SHIFT: u8 = 22;
const DEFAULT_LINEAR_BACKWARD_LEARNING_RATE: i32 = 1;
const DEFAULT_LINEAR_BACKWARD_LEARNING_RATE_SHIFT: u8 = 22;
const DEFAULT_BYTE_SOFTMAX_EPOCHS: usize = 1;
const DEFAULT_BYTE_SOFTMAX_SEQ_LEN: usize = 8;
const DEFAULT_BYTE_SOFTMAX_STRIDE: usize = 1;
const DEFAULT_BYTE_SOFTMAX_MAX_WINDOWS: usize = 128;
const DEFAULT_BYTE_SOFTMAX_LEARNING_RATE: i32 = 1;
const DEFAULT_BYTE_SOFTMAX_LEARNING_RATE_SHIFT: u8 = 22;
const DEFAULT_BYTE_EMBED_SOFTMAX_EPOCHS: usize = 1;
const DEFAULT_BYTE_EMBED_SOFTMAX_SEQ_LEN: usize = 8;
const DEFAULT_BYTE_EMBED_SOFTMAX_STRIDE: usize = 1;
const DEFAULT_BYTE_EMBED_SOFTMAX_MAX_WINDOWS: usize = 128;
const DEFAULT_BYTE_EMBED_SOFTMAX_LEARNING_RATE: i32 = 1;
const DEFAULT_BYTE_EMBED_SOFTMAX_HEAD_LEARNING_RATE_SHIFT: u8 = 17;
const DEFAULT_BYTE_EMBED_SOFTMAX_EMBEDDING_LEARNING_RATE_SHIFT: u8 = 0;
const DEFAULT_MINI_TRANSFORMER_EPOCHS: usize = 1;
const DEFAULT_MINI_TRANSFORMER_SEQ_LEN: usize = 4;
const DEFAULT_MINI_TRANSFORMER_STRIDE: usize = 1;
const DEFAULT_MINI_TRANSFORMER_MAX_WINDOWS: usize = 64;
const DEFAULT_MINI_TRANSFORMER_LEARNING_RATE: i32 = 1;
const DEFAULT_MINI_TRANSFORMER_HEAD_LEARNING_RATE_SHIFT: u8 = 18;
const DEFAULT_MINI_TRANSFORMER_MLP_LEARNING_RATE_SHIFT: u8 = 16;
const DEFAULT_MINI_TRANSFORMER_ATTENTION_LEARNING_RATE_SHIFT: u8 = 24;
const DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT: u8 = 18;
const BASE2_SOFTMAX_LN2_Q15: i32 = 22_713;
const LOGIT_SCALES: [FixedScale; VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; VOCAB];
const BYTE_LOGIT_SCALES: [FixedScale; BYTE_VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; BYTE_VOCAB];
const BYTE_EMBED_LOGIT_SCALES: [FixedScale; BYTE_VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; BYTE_VOCAB];
const BYTE_EMBED_GRAD_FEATURE_SCALES: [FixedScale; BYTE_EMBED_D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; BYTE_EMBED_D_MODEL];
const MINI_TRANSFORMER_D_MODEL_SCALES: [FixedScale; MINI_TRANSFORMER_D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    MINI_TRANSFORMER_D_MODEL];
const MINI_TRANSFORMER_HIDDEN_SCALES: [FixedScale; MINI_TRANSFORMER_HIDDEN_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    MINI_TRANSFORMER_HIDDEN_DIM];
const MINI_TRANSFORMER_OUTPUT_SCALES: [FixedScale; BYTE_VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; BYTE_VOCAB];
const MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES: [FixedScale; MINI_TRANSFORMER_D_MODEL] =
    [FixedScale {
        multiplier: 1,
        right_shift: 0,
    }; MINI_TRANSFORMER_D_MODEL];
const LINEAR_BACKWARD_INPUT_Q15: [i16; LINEAR_BACKWARD_INPUT_DIM] = [4096, -2048, 1024, -512];
const LINEAR_BACKWARD_GRAD_OUTPUT_Q15: [i16; LINEAR_BACKWARD_OUTPUT_DIM] = [8192, -4096, 2048];
const LINEAR_BACKWARD_INITIAL_WEIGHTS: [i8; LINEAR_BACKWARD_INPUT_DIM
    * LINEAR_BACKWARD_OUTPUT_DIM] = [
    3, -2, 1, 0, //
    -1, 4, 0, 2, //
    2, 1, -3, 1,
];
const LINEAR_BACKWARD_FORWARD_SCALES: [FixedScale; LINEAR_BACKWARD_OUTPUT_DIM] = [
    FixedScale {
        multiplier: 1,
        right_shift: 0,
    },
    FixedScale {
        multiplier: 3,
        right_shift: 1,
    },
    FixedScale {
        multiplier: 1,
        right_shift: 2,
    },
];
const LINEAR_BACKWARD_GRAD_INPUT_SCALES: [FixedScale; LINEAR_BACKWARD_INPUT_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 3,
};
    LINEAR_BACKWARD_INPUT_DIM];
const GATED_MLP_BACKWARD_INPUT_Q15: [i16; GATED_MLP_BACKWARD_D_MODEL] = [4096, -2048];
const GATED_MLP_BACKWARD_GRAD_OUTPUT_Q15: [i16; GATED_MLP_BACKWARD_D_MODEL] = [8192, -4096];
const GATED_MLP_BACKWARD_UP_INITIAL: [i8; GATED_MLP_BACKWARD_D_MODEL
    * GATED_MLP_BACKWARD_HIDDEN_DIM] = [
    1, 0, //
    0, 1, //
    1, 1, //
    -1, 1,
];
const GATED_MLP_BACKWARD_GATE_INITIAL: [i8; GATED_MLP_BACKWARD_D_MODEL
    * GATED_MLP_BACKWARD_HIDDEN_DIM] = GATED_MLP_BACKWARD_UP_INITIAL;
const GATED_MLP_BACKWARD_DOWN_INITIAL: [i8; GATED_MLP_BACKWARD_HIDDEN_DIM
    * GATED_MLP_BACKWARD_D_MODEL] = [
    1, 0, 1, -1, //
    0, 1, 1, 1,
];
const GATED_MLP_BACKWARD_D_MODEL_SCALES: [FixedScale; GATED_MLP_BACKWARD_D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    GATED_MLP_BACKWARD_D_MODEL];
const GATED_MLP_BACKWARD_HIDDEN_SCALES: [FixedScale; GATED_MLP_BACKWARD_HIDDEN_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    GATED_MLP_BACKWARD_HIDDEN_DIM];
const FEATURES_Q15: [[i16; D_MODEL]; VOCAB] = [
    [4096, 0, 0, 0, 2048, -1024, 512, 256],
    [0, 4096, 0, 0, -1024, 2048, -512, 256],
    [0, 0, 4096, 0, 512, -512, 2048, -1024],
    [0, 0, 0, 4096, -512, 512, -1024, 2048],
];
const TRAINING_PAIRS: [(usize, usize); 8] = [
    (0, 1),
    (1, 0),
    (0, 1),
    (1, 0),
    (2, 3),
    (3, 2),
    (2, 3),
    (3, 2),
];
const KNOWN_NON_CLAIMS: [&str; 4] = [
    "not_transformer_backprop_yet",
    "does_not_update_attention_or_mlp_weights_yet",
    "does_not_claim_language_model_quality",
    "does_not_prove_general_training_convergence",
];
const SOFTMAX_KNOWN_NON_CLAIMS: [&str; 4] = [
    "not_full_transformer_backprop_yet",
    "does_not_update_attention_or_mlp_weights_yet",
    "does_not_update_gated_mlp_weights_yet",
    "does_not_claim_language_model_quality",
];
const LINEAR_BACKWARD_KNOWN_NON_CLAIMS: [&str; 4] = [
    "single_linear_layer_only",
    "does_not_backpropagate_through_attention_yet",
    "does_not_backpropagate_through_rmsnorm_yet",
    "does_not_claim_language_model_quality",
];
const GATED_MLP_BACKWARD_KNOWN_NON_CLAIMS: [&str; 4] = [
    "single_gated_mlp_only",
    "does_not_backpropagate_through_attention_yet",
    "does_not_backpropagate_through_rmsnorm_yet",
    "does_not_claim_language_model_quality",
];
const BYTE_SOFTMAX_KNOWN_NON_CLAIMS: [&str; 5] = [
    "output_head_only",
    "features_are_bias_plus_last_byte_one_hot",
    "does_not_update_transformer_weights_yet",
    "does_not_claim_language_model_quality",
    "not_full_wiki_bard_training_yet",
];
const BYTE_GENERATION_KNOWN_NON_CLAIMS: [&str; 4] = [
    "baseline_byte_model_only",
    "not_transformer_generation_yet",
    "greedy_decoding_only",
    "does_not_claim_language_model_quality",
];
const BYTE_EMBED_SOFTMAX_KNOWN_NON_CLAIMS: [&str; 5] = [
    "learned_embedding_output_head_only",
    "context_state_is_mean_byte_embedding",
    "does_not_update_transformer_weights_yet",
    "does_not_claim_language_model_quality",
    "not_full_wiki_bard_training_yet",
];
const BYTE_EMBED_GENERATION_KNOWN_NON_CLAIMS: [&str; 4] = [
    "baseline_embedding_byte_model_only",
    "not_transformer_generation_yet",
    "greedy_decoding_only",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS: [&str; 4] = [
    "single_mini_transformer_block_only",
    "embedding_table_forward_only_no_embedding_update_yet",
    "greedy_decoding_only",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS: [&str; 5] = [
    "single_mini_transformer_block_only",
    "embedding_table_forward_only_no_embedding_update_yet",
    "single_head_attention_only",
    "does_not_backpropagate_through_rmsnorm_yet",
    "does_not_claim_language_model_quality",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: i8,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_EPOCHS,
            learning_rate: DEFAULT_LEARNING_RATE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxTrainConfig {
    pub epochs: usize,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
}

impl Default for SoftmaxTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_EPOCHS,
            learning_rate: DEFAULT_SOFTMAX_LEARNING_RATE,
            learning_rate_shift: DEFAULT_SOFTMAX_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearBackwardConfig {
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
}

impl Default for LinearBackwardConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LINEAR_BACKWARD_LEARNING_RATE,
            learning_rate_shift: DEFAULT_LINEAR_BACKWARD_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSoftmaxTrainConfig {
    pub epochs: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
}

impl Default for ByteSoftmaxTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_BYTE_SOFTMAX_EPOCHS,
            seq_len: DEFAULT_BYTE_SOFTMAX_SEQ_LEN,
            stride: DEFAULT_BYTE_SOFTMAX_STRIDE,
            max_windows: Some(DEFAULT_BYTE_SOFTMAX_MAX_WINDOWS),
            learning_rate: DEFAULT_BYTE_SOFTMAX_LEARNING_RATE,
            learning_rate_shift: DEFAULT_BYTE_SOFTMAX_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteEmbedSoftmaxTrainConfig {
    pub epochs: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub learning_rate: i32,
    pub head_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
}

impl Default for ByteEmbedSoftmaxTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_BYTE_EMBED_SOFTMAX_EPOCHS,
            seq_len: DEFAULT_BYTE_EMBED_SOFTMAX_SEQ_LEN,
            stride: DEFAULT_BYTE_EMBED_SOFTMAX_STRIDE,
            max_windows: Some(DEFAULT_BYTE_EMBED_SOFTMAX_MAX_WINDOWS),
            learning_rate: DEFAULT_BYTE_EMBED_SOFTMAX_LEARNING_RATE,
            head_learning_rate_shift: DEFAULT_BYTE_EMBED_SOFTMAX_HEAD_LEARNING_RATE_SHIFT,
            embedding_learning_rate_shift: DEFAULT_BYTE_EMBED_SOFTMAX_EMBEDDING_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainConfig {
    pub epochs: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub learning_rate: i32,
    pub output_learning_rate_shift: u8,
    pub mlp_learning_rate_shift: u8,
    pub attention_learning_rate_shift: u8,
    pub attention_qk_learning_rate_shift: u8,
}

impl Default for MiniTransformerMlpTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_MINI_TRANSFORMER_EPOCHS,
            seq_len: DEFAULT_MINI_TRANSFORMER_SEQ_LEN,
            stride: DEFAULT_MINI_TRANSFORMER_STRIDE,
            max_windows: Some(DEFAULT_MINI_TRANSFORMER_MAX_WINDOWS),
            learning_rate: DEFAULT_MINI_TRANSFORMER_LEARNING_RATE,
            output_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_HEAD_LEARNING_RATE_SHIFT,
            mlp_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_MLP_LEARNING_RATE_SHIFT,
            attention_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_ATTENTION_LEARNING_RATE_SHIFT,
            attention_qk_learning_rate_shift:
                DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingTrace {
    pub config: TrainConfig,
    pub samples: usize,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub updates: usize,
    pub gradient_saturation_count: usize,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<TrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftmaxTrainingTrace {
    pub config: SoftmaxTrainConfig,
    pub samples: usize,
    pub examined_samples: usize,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub updates: usize,
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<SoftmaxTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearBackwardTrace {
    pub config: LinearBackwardConfig,
    pub input_q15: [i16; LINEAR_BACKWARD_INPUT_DIM],
    pub grad_output_q15: [i16; LINEAR_BACKWARD_OUTPUT_DIM],
    pub scaled_grad_output_i32: [i32; LINEAR_BACKWARD_OUTPUT_DIM],
    pub grad_input_q15: [i16; LINEAR_BACKWARD_INPUT_DIM],
    pub output_before_q15: [i16; LINEAR_BACKWARD_OUTPUT_DIM],
    pub output_after_q15: [i16; LINEAR_BACKWARD_OUTPUT_DIM],
    pub weights_before_i8: [i8; LINEAR_BACKWARD_INPUT_DIM * LINEAR_BACKWARD_OUTPUT_DIM],
    pub weights_after_i8: [i8; LINEAR_BACKWARD_INPUT_DIM * LINEAR_BACKWARD_OUTPUT_DIM],
    pub update_stats: LinearWeightUpdateStats,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub output_hash_before: u64,
    pub output_hash_after: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedMlpBackwardTrace {
    pub config: LinearBackwardConfig,
    pub input_q15: [i16; GATED_MLP_BACKWARD_D_MODEL],
    pub grad_output_q15: [i16; GATED_MLP_BACKWARD_D_MODEL],
    pub forward_up_q15: [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub forward_gate_q15: [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub forward_gated_q15: [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub output_before_q15: [i16; GATED_MLP_BACKWARD_D_MODEL],
    pub output_after_q15: [i16; GATED_MLP_BACKWARD_D_MODEL],
    pub up_before_i8: [i8; GATED_MLP_BACKWARD_D_MODEL * GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub up_after_i8: [i8; GATED_MLP_BACKWARD_D_MODEL * GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub gate_before_i8: [i8; GATED_MLP_BACKWARD_D_MODEL * GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub gate_after_i8: [i8; GATED_MLP_BACKWARD_D_MODEL * GATED_MLP_BACKWARD_HIDDEN_DIM],
    pub down_before_i8: [i8; GATED_MLP_BACKWARD_HIDDEN_DIM * GATED_MLP_BACKWARD_D_MODEL],
    pub down_after_i8: [i8; GATED_MLP_BACKWARD_HIDDEN_DIM * GATED_MLP_BACKWARD_D_MODEL],
    pub update_stats: GatedMlpWeightUpdateStats,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub output_hash_before: u64,
    pub output_hash_after: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSoftmaxTrainingTrace {
    pub config: ByteSoftmaxTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<ByteSoftmaxTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSoftmaxTrainingRun {
    pub trace: ByteSoftmaxTrainingTrace,
    pub model: ByteSoftmaxModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEmbedSoftmaxTrainingTrace {
    pub config: ByteEmbedSoftmaxTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub initial_embedding_hash: u64,
    pub final_embedding_hash: u64,
    pub initial_weight_hash: u64,
    pub final_weight_hash: u64,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub gradient_saturation_count: usize,
    pub embedding_saturation_count: usize,
    pub zero_delta_count: usize,
    pub embedding_zero_delta_count: usize,
    pub weight_delta_l1: u64,
    pub embedding_delta_l1: u64,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<ByteEmbedSoftmaxTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEmbedSoftmaxTrainingRun {
    pub trace: ByteEmbedSoftmaxTrainingTrace,
    pub model: ByteEmbedSoftmaxModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingTrace {
    pub config: MiniTransformerMlpTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
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
    pub attention_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub output_head_zero_delta_count: usize,
    pub mlp_zero_delta_count: usize,
    pub attention_zero_delta_count: usize,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<MiniTransformerMlpTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingRun {
    pub trace: MiniTransformerMlpTrainingTrace,
    pub model: MiniTransformerMlpModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSoftmaxModel {
    pub weights: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEmbedSoftmaxModel {
    pub seq_len: usize,
    pub embeddings: Vec<i16>,
    pub output_weights: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpModel {
    pub context_seq_len: usize,
    pub embeddings: Vec<i16>,
    pub q_weights: Vec<i8>,
    pub k_weights: Vec<i8>,
    pub v_weights: Vec<i8>,
    pub o_weights: Vec<i8>,
    pub up_weights: Vec<i8>,
    pub gate_weights: Vec<i8>,
    pub down_weights: Vec<i8>,
    pub output_weights: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGenerationConfig {
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGenerationTrace {
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGenerationStepTrace {
    pub step_index: usize,
    pub input_token: u8,
    pub predicted_token: u8,
    pub predicted_logit_q8: i32,
    pub predicted_probability_q15: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEmbedGenerationTrace {
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub output_weight_hash: u64,
    pub context_seq_len: usize,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerGenerationTrace {
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
    pub context_seq_len: usize,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub sample_index: usize,
    pub input_id: usize,
    pub target_id: usize,
    pub predicted_id: usize,
    pub total_error_before: usize,
    pub total_error_after: usize,
    pub error_delta_i32: i32,
    pub weight_hash_before: u64,
    pub weight_hash_after: u64,
    pub gradient_saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftmaxTrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub sample_index: usize,
    pub input_id: usize,
    pub target_id: usize,
    pub predicted_id_before: usize,
    pub predicted_id_after: usize,
    pub logits_q8_before: [i32; VOCAB],
    pub probabilities_q15_before: [i16; VOCAB],
    pub gradient_q15: [i32; VOCAB],
    pub softmax_sum_q15_before: u64,
    pub total_error_before: usize,
    pub total_error_after: usize,
    pub error_delta_i32: i32,
    pub probability_error_before_q15: usize,
    pub probability_error_after_q15: usize,
    pub probability_error_delta_i32: i32,
    pub weight_hash_before: u64,
    pub weight_hash_after: u64,
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSoftmaxTrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub window_index: usize,
    pub window_start: usize,
    pub last_token: u8,
    pub target_token: u8,
    pub predicted_token_before: u8,
    pub predicted_token_after: u8,
    pub target_probability_before_q15: i16,
    pub target_probability_after_q15: i16,
    pub weight_hash_before: u64,
    pub weight_hash_after: u64,
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEmbedSoftmaxTrainingStepTrace {
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
    pub embedding_hash_before: u64,
    pub embedding_hash_after: u64,
    pub weight_hash_before: u64,
    pub weight_hash_after: u64,
    pub gradient_saturation_count: usize,
    pub embedding_saturation_count: usize,
    pub zero_delta_count: usize,
    pub embedding_zero_delta_count: usize,
    pub weight_delta_l1: u64,
    pub embedding_delta_l1: u64,
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
    pub output_head_hash_before: u64,
    pub output_head_hash_after: u64,
    pub mlp_hash_before: u64,
    pub mlp_hash_after: u64,
    pub attention_hash_before: u64,
    pub attention_hash_after: u64,
    pub output_head_saturation_count: usize,
    pub mlp_saturation_count: usize,
    pub attention_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub output_head_zero_delta_count: usize,
    pub mlp_zero_delta_count: usize,
    pub attention_zero_delta_count: usize,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerMlpForwardCache {
    embedding_output: Vec<i16>,
    attention_q: Vec<i16>,
    attention_k: Vec<i16>,
    attention_v: Vec<i16>,
    attention_context: Vec<i16>,
    attention_probabilities_q15: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_up: Vec<i16>,
    mlp_gate: Vec<i16>,
    mlp_gated: Vec<i16>,
    mlp_output: Vec<i16>,
    block_output: Vec<i16>,
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
    residual_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerAttentionWeightUpdateStats {
    q: LinearWeightUpdateStats,
    k: LinearWeightUpdateStats,
    v: LinearWeightUpdateStats,
    o: LinearWeightUpdateStats,
    gradient_saturation_count: usize,
    zero_delta_count: usize,
    weight_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    InvalidConfig,
    InvalidModel(&'static str),
    CoreRejected(&'static str),
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid training config"),
            Self::InvalidModel(message) => write!(f, "invalid model artifact: {message}"),
            Self::CoreRejected(stage) => write!(f, "nsrl-core rejected training stage: {stage}"),
        }
    }
}

impl std::error::Error for TrainError {}

pub fn run_training_smoke(config: TrainConfig) -> Result<TrainingTrace, TrainError> {
    if config.epochs == 0 || config.learning_rate <= 0 {
        return Err(TrainError::InvalidConfig);
    }

    let mut weights = [0_i8; VOCAB * D_MODEL];
    let initial_weight_hash = hash_i8_slice(&weights);
    let initial_total_error = total_error(&weights)?;
    let initial_mistakes = count_mistakes(&weights)?;
    let mut updates = 0_usize;
    let mut gradient_saturation_count = 0_usize;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (sample_index, &(input_id, target_id)) in TRAINING_PAIRS.iter().enumerate() {
            let logits = logits_for(&weights, input_id)?;
            let predicted_id = argmax(&logits);
            if predicted_id == target_id {
                continue;
            }

            let total_error_before = total_error(&weights)?;
            let weight_hash_before = hash_i8_slice(&weights);
            let target_saturation_count =
                apply_perceptron_update(&mut weights, input_id, target_id, config.learning_rate, 1);
            let predicted_saturation_count = apply_perceptron_update(
                &mut weights,
                input_id,
                predicted_id,
                config.learning_rate,
                -1,
            );
            let step_saturation_count = target_saturation_count + predicted_saturation_count;
            gradient_saturation_count += step_saturation_count;
            updates += 1;
            let total_error_after = total_error(&weights)?;
            let weight_hash_after = hash_i8_slice(&weights);
            steps.push(TrainingStepTrace {
                update_index: updates,
                epoch,
                sample_index,
                input_id,
                target_id,
                predicted_id,
                total_error_before,
                total_error_after,
                error_delta_i32: total_error_after as i32 - total_error_before as i32,
                weight_hash_before,
                weight_hash_after,
                gradient_saturation_count: step_saturation_count,
            });
        }
    }

    let final_total_error = total_error(&weights)?;
    let final_mistakes = count_mistakes(&weights)?;
    let final_correct = TRAINING_PAIRS.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / TRAINING_PAIRS.len();
    let final_logits_hash = hash_logits(&weights)?;

    Ok(TrainingTrace {
        config,
        samples: TRAINING_PAIRS.len(),
        initial_weight_hash,
        final_weight_hash: hash_i8_slice(&weights),
        initial_total_error,
        final_total_error,
        initial_mistakes,
        final_mistakes,
        updates,
        gradient_saturation_count,
        final_accuracy_per_mille,
        final_logits_hash,
        steps,
    })
}

pub fn run_softmax_training(
    config: SoftmaxTrainConfig,
) -> Result<SoftmaxTrainingTrace, TrainError> {
    if config.epochs == 0
        || config.learning_rate <= 0
        || config.learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut weights = [0_i8; VOCAB * D_MODEL];
    let initial_weight_hash = hash_i8_slice(&weights);
    let initial_total_error = total_error(&weights)?;
    let initial_probability_error_q15 = total_probability_error_q15(&weights)?;
    let initial_mistakes = count_mistakes(&weights)?;
    let mut updates = 0_usize;
    let mut examined_samples = 0_usize;
    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    'epochs: for epoch in 0..config.epochs {
        for (sample_index, &(input_id, target_id)) in TRAINING_PAIRS.iter().enumerate() {
            examined_samples += 1;
            let row = softmax_row_for(&weights, input_id)?;
            let predicted_id_before = argmax_i32(&row.logits_q8);
            let gradient_q15 = softmax_gradient_q15(&row.probabilities_q15, target_id);
            let total_error_before = total_error(&weights)?;
            let probability_error_before_q15 = total_probability_error_q15(&weights)?;
            let weight_hash_before = hash_i8_slice(&weights);
            let update = apply_softmax_output_head_update(
                &mut weights,
                input_id,
                &gradient_q15,
                config.learning_rate,
                config.learning_rate_shift,
            );
            updates += 1;
            gradient_saturation_count += update.gradient_saturation_count;
            zero_delta_count += update.zero_delta_count;
            weight_delta_l1 = weight_delta_l1.saturating_add(update.weight_delta_l1);
            let total_error_after = total_error(&weights)?;
            let probability_error_after_q15 = total_probability_error_q15(&weights)?;
            let after_row = softmax_row_for(&weights, input_id)?;
            let predicted_id_after = argmax_i32(&after_row.logits_q8);
            let weight_hash_after = hash_i8_slice(&weights);

            steps.push(SoftmaxTrainingStepTrace {
                update_index: updates,
                epoch,
                sample_index,
                input_id,
                target_id,
                predicted_id_before,
                predicted_id_after,
                logits_q8_before: row.logits_q8,
                probabilities_q15_before: row.probabilities_q15,
                gradient_q15,
                softmax_sum_q15_before: row.softmax_sum_q15,
                total_error_before,
                total_error_after,
                error_delta_i32: total_error_after as i32 - total_error_before as i32,
                probability_error_before_q15,
                probability_error_after_q15,
                probability_error_delta_i32: probability_error_after_q15 as i32
                    - probability_error_before_q15 as i32,
                weight_hash_before,
                weight_hash_after,
                gradient_saturation_count: update.gradient_saturation_count,
                zero_delta_count: update.zero_delta_count,
                weight_delta_l1: update.weight_delta_l1,
            });

            if total_error_after == 0 {
                break 'epochs;
            }
        }
    }

    let final_total_error = total_error(&weights)?;
    let final_probability_error_q15 = total_probability_error_q15(&weights)?;
    let final_mistakes = count_mistakes(&weights)?;
    let final_correct = TRAINING_PAIRS.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / TRAINING_PAIRS.len();
    let final_logits_hash = hash_logits(&weights)?;

    Ok(SoftmaxTrainingTrace {
        config,
        samples: TRAINING_PAIRS.len(),
        examined_samples,
        initial_weight_hash,
        final_weight_hash: hash_i8_slice(&weights),
        initial_total_error,
        final_total_error,
        initial_probability_error_q15,
        final_probability_error_q15,
        initial_mistakes,
        final_mistakes,
        updates,
        gradient_saturation_count,
        zero_delta_count,
        weight_delta_l1,
        final_accuracy_per_mille,
        final_logits_hash,
        steps,
    })
}

pub fn run_linear_backward_smoke(
    config: LinearBackwardConfig,
) -> Result<LinearBackwardTrace, TrainError> {
    if config.learning_rate <= 0 || config.learning_rate_shift > MAX_RIGHT_SHIFT {
        return Err(TrainError::InvalidConfig);
    }

    let mut weights = LINEAR_BACKWARD_INITIAL_WEIGHTS;
    let weights_before_i8 = weights;
    let initial_weight_hash = hash_i8_slice(&weights);
    let output_before_q15 = linear_backward_output_for(&weights)?;
    let output_hash_before = hash_i16_slice(&output_before_q15);

    let mut scaled_grad_output_i32 = [0_i32; LINEAR_BACKWARD_OUTPUT_DIM];
    let mut grad_input_q15 = [0_i16; LINEAR_BACKWARD_INPUT_DIM];
    linear_backward_input_i16_i8_i16_per_channel_checked(
        &LINEAR_BACKWARD_GRAD_OUTPUT_Q15,
        LinearBackwardInputI16I8Params {
            weights: &weights,
            forward_scales: &LINEAR_BACKWARD_FORWARD_SCALES,
            grad_input_scales: &LINEAR_BACKWARD_GRAD_INPUT_SCALES,
            input_dim: LINEAR_BACKWARD_INPUT_DIM,
            output_dim: LINEAR_BACKWARD_OUTPUT_DIM,
        },
        LinearBackwardInputWorkspace {
            scaled_grad_output: &mut scaled_grad_output_i32,
        },
        &mut grad_input_q15,
    )
    .ok_or(TrainError::CoreRejected("linear_backward_input"))?;

    let mut update_scaled_grad_output = [0_i32; LINEAR_BACKWARD_OUTPUT_DIM];
    let update_stats = linear_backward_weight_update_i8_checked(
        &LINEAR_BACKWARD_INPUT_Q15,
        &LINEAR_BACKWARD_GRAD_OUTPUT_Q15,
        &mut weights,
        LinearBackwardWeightUpdateI8Params {
            forward_scales: &LINEAR_BACKWARD_FORWARD_SCALES,
            input_dim: LINEAR_BACKWARD_INPUT_DIM,
            output_dim: LINEAR_BACKWARD_OUTPUT_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.learning_rate_shift,
        },
        LinearBackwardWeightUpdateWorkspace {
            scaled_grad_output: &mut update_scaled_grad_output,
        },
    )
    .ok_or(TrainError::CoreRejected("linear_backward_weight_update"))?;

    if update_scaled_grad_output != scaled_grad_output_i32 {
        return Err(TrainError::CoreRejected(
            "linear_backward_prescale_replay_mismatch",
        ));
    }

    let output_after_q15 = linear_backward_output_for(&weights)?;
    let output_hash_after = hash_i16_slice(&output_after_q15);

    Ok(LinearBackwardTrace {
        config,
        input_q15: LINEAR_BACKWARD_INPUT_Q15,
        grad_output_q15: LINEAR_BACKWARD_GRAD_OUTPUT_Q15,
        scaled_grad_output_i32,
        grad_input_q15,
        output_before_q15,
        output_after_q15,
        weights_before_i8,
        weights_after_i8: weights,
        update_stats,
        initial_weight_hash,
        final_weight_hash: hash_i8_slice(&weights),
        output_hash_before,
        output_hash_after,
    })
}

pub fn run_gated_mlp_backward_smoke(
    config: LinearBackwardConfig,
) -> Result<GatedMlpBackwardTrace, TrainError> {
    if config.learning_rate <= 0 || config.learning_rate_shift > MAX_RIGHT_SHIFT {
        return Err(TrainError::InvalidConfig);
    }

    let mut up_weights = GATED_MLP_BACKWARD_UP_INITIAL;
    let mut gate_weights = GATED_MLP_BACKWARD_GATE_INITIAL;
    let mut down_weights = GATED_MLP_BACKWARD_DOWN_INITIAL;
    let up_before_i8 = up_weights;
    let gate_before_i8 = gate_weights;
    let down_before_i8 = down_weights;
    let initial_weight_hash = hash_three_i8_slices(&up_weights, &gate_weights, &down_weights);
    let (forward_up_q15, forward_gate_q15, forward_gated_q15, output_before_q15) =
        gated_mlp_backward_forward_for(&up_weights, &gate_weights, &down_weights)?;
    let output_hash_before = hash_i16_slice(&output_before_q15);

    let mut scaled_grad_output = [0_i32; GATED_MLP_BACKWARD_SCALED_WORKSPACE_DIM];
    let mut grad_gated = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let mut grad_up = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let mut grad_gate = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let update_stats = gated_mlp_backward_weight_update_i8_checked(
        &GATED_MLP_BACKWARD_INPUT_Q15,
        &GATED_MLP_BACKWARD_GRAD_OUTPUT_Q15,
        &forward_up_q15,
        &forward_gate_q15,
        &forward_gated_q15,
        &mut up_weights,
        &mut gate_weights,
        &mut down_weights,
        GatedMlpWeightUpdateParams {
            up_scales: &GATED_MLP_BACKWARD_HIDDEN_SCALES,
            gate_scales: &GATED_MLP_BACKWARD_HIDDEN_SCALES,
            down_scales: &GATED_MLP_BACKWARD_D_MODEL_SCALES,
            down_to_hidden_scales: &GATED_MLP_BACKWARD_HIDDEN_SCALES,
            seq_len: GATED_MLP_BACKWARD_SEQ_LEN,
            d_model: GATED_MLP_BACKWARD_D_MODEL,
            hidden_dim: GATED_MLP_BACKWARD_HIDDEN_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.learning_rate_shift,
        },
        GatedMlpWeightUpdateWorkspace {
            scaled_grad_output: &mut scaled_grad_output,
            grad_gated: &mut grad_gated,
            grad_up: &mut grad_up,
            grad_gate: &mut grad_gate,
        },
    )
    .ok_or(TrainError::CoreRejected("gated_mlp_backward_weight_update"))?;

    let (_, _, _, output_after_q15) =
        gated_mlp_backward_forward_for(&up_weights, &gate_weights, &down_weights)?;
    let output_hash_after = hash_i16_slice(&output_after_q15);

    Ok(GatedMlpBackwardTrace {
        config,
        input_q15: GATED_MLP_BACKWARD_INPUT_Q15,
        grad_output_q15: GATED_MLP_BACKWARD_GRAD_OUTPUT_Q15,
        forward_up_q15,
        forward_gate_q15,
        forward_gated_q15,
        output_before_q15,
        output_after_q15,
        up_before_i8,
        up_after_i8: up_weights,
        gate_before_i8,
        gate_after_i8: gate_weights,
        down_before_i8,
        down_after_i8: down_weights,
        update_stats,
        initial_weight_hash,
        final_weight_hash: hash_three_i8_slices(&up_weights, &gate_weights, &down_weights),
        output_hash_before,
        output_hash_after,
    })
}

pub fn run_byte_softmax_training(
    tokens: &[u8],
    config: ByteSoftmaxTrainConfig,
) -> Result<ByteSoftmaxTrainingTrace, TrainError> {
    Ok(run_byte_softmax_training_with_model(tokens, config)?.trace)
}

pub fn run_byte_softmax_training_with_model(
    tokens: &[u8],
    config: ByteSoftmaxTrainConfig,
) -> Result<ByteSoftmaxTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.learning_rate <= 0
        || config.learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let starts = byte_window_starts(
        tokens.len(),
        config.seq_len,
        config.stride,
        config.max_windows,
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut weights = vec![0_i8; BYTE_VOCAB * BYTE_D_MODEL];
    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_byte_windows(tokens, config, &starts);
    let initial_weight_hash = hash_i8_slice(&weights);
    let initial_total_error = byte_total_error(tokens, &starts, &weights, config.seq_len)?;
    let initial_probability_error_q15 =
        byte_total_probability_error_q15(tokens, &starts, &weights, config.seq_len)?;
    let initial_mistakes = initial_total_error;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (window_index, &window_start) in starts.iter().enumerate() {
            examined_windows += 1;
            let features = byte_features_q15(tokens, window_start, config.seq_len);
            let target_token = tokens[window_start + config.seq_len];
            let row = byte_softmax_row_for(&weights, &features)?;
            let predicted_token_before = byte_argmax_i32(&row.logits_q8);
            let gradient_q15 = byte_softmax_gradient_q15(&row.probabilities_q15, target_token);
            let weight_hash_before = hash_i8_slice(&weights);
            let update = apply_byte_softmax_output_head_update(
                &mut weights,
                &features,
                &gradient_q15,
                config.learning_rate,
                config.learning_rate_shift,
            );
            updates += 1;
            gradient_saturation_count += update.gradient_saturation_count;
            zero_delta_count += update.zero_delta_count;
            weight_delta_l1 = weight_delta_l1.saturating_add(update.weight_delta_l1);
            let after_row = byte_softmax_row_for(&weights, &features)?;
            let predicted_token_after = byte_argmax_i32(&after_row.logits_q8);
            let weight_hash_after = hash_i8_slice(&weights);

            steps.push(ByteSoftmaxTrainingStepTrace {
                update_index: updates,
                epoch,
                window_index,
                window_start,
                last_token: tokens[window_start + config.seq_len - 1],
                target_token,
                predicted_token_before,
                predicted_token_after,
                target_probability_before_q15: row.probabilities_q15[usize::from(target_token)],
                target_probability_after_q15: after_row.probabilities_q15
                    [usize::from(target_token)],
                weight_hash_before,
                weight_hash_after,
                gradient_saturation_count: update.gradient_saturation_count,
                zero_delta_count: update.zero_delta_count,
                weight_delta_l1: update.weight_delta_l1,
            });
        }
    }

    let final_total_error = byte_total_error(tokens, &starts, &weights, config.seq_len)?;
    let final_probability_error_q15 =
        byte_total_probability_error_q15(tokens, &starts, &weights, config.seq_len)?;
    let final_mistakes = final_total_error;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = hash_byte_logits(tokens, &starts, &weights, config.seq_len)?;
    let model = ByteSoftmaxModel { weights };
    let final_weight_hash = model.weight_hash();

    let trace = ByteSoftmaxTrainingTrace {
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        initial_weight_hash,
        final_weight_hash,
        initial_total_error,
        final_total_error,
        initial_probability_error_q15,
        final_probability_error_q15,
        initial_mistakes,
        final_mistakes,
        gradient_saturation_count,
        zero_delta_count,
        weight_delta_l1,
        final_accuracy_per_mille,
        final_logits_hash,
        steps,
    };

    Ok(ByteSoftmaxTrainingRun { trace, model })
}

pub fn run_byte_embed_softmax_training(
    tokens: &[u8],
    config: ByteEmbedSoftmaxTrainConfig,
) -> Result<ByteEmbedSoftmaxTrainingTrace, TrainError> {
    Ok(run_byte_embed_softmax_training_with_model(tokens, config)?.trace)
}

pub fn run_byte_embed_softmax_training_with_model(
    tokens: &[u8],
    config: ByteEmbedSoftmaxTrainConfig,
) -> Result<ByteEmbedSoftmaxTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || !config.seq_len.is_power_of_two()
        || config.stride == 0
        || config.learning_rate <= 0
        || config.head_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let seq_shift = byte_embed_seq_shift(config.seq_len)?;
    let Some(total_embedding_shift) = config.embedding_learning_rate_shift.checked_add(seq_shift)
    else {
        return Err(TrainError::InvalidConfig);
    };
    if total_embedding_shift > MAX_RIGHT_SHIFT {
        return Err(TrainError::InvalidConfig);
    }

    let starts = byte_window_starts(
        tokens.len(),
        config.seq_len,
        config.stride,
        config.max_windows,
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_byte_embed_windows(tokens, config, &starts);
    let mut embeddings = initial_byte_embed_embeddings();
    let mut output_weights = initial_byte_embed_output_weights();
    let initial_embedding_hash = hash_i16_slice(&embeddings);
    let initial_weight_hash = hash_i8_slice(&output_weights);
    let initial_total_error = byte_embed_total_error(
        tokens,
        &starts,
        &embeddings,
        &output_weights,
        config.seq_len,
    )?;
    let initial_probability_error_q15 = byte_embed_total_probability_error_q15(
        tokens,
        &starts,
        &embeddings,
        &output_weights,
        config.seq_len,
    )?;
    let initial_mistakes = initial_total_error;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut gradient_saturation_count = 0_usize;
    let mut embedding_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut embedding_zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;
    let mut embedding_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (window_index, &window_start) in starts.iter().enumerate() {
            examined_windows += 1;
            let features =
                byte_embed_features_q15(&embeddings, tokens, window_start, config.seq_len)?;
            let target_token = tokens[window_start + config.seq_len];
            let row = byte_embed_softmax_row_for(&output_weights, &features)?;
            let predicted_token_before = byte_argmax_i32(&row.logits_q8);
            let gradient_q15 = byte_softmax_gradient_q15(&row.probabilities_q15, target_token);
            let grad_output_q15 = byte_gradient_i32_to_i16(&gradient_q15);
            let embedding_hash_before = hash_i16_slice(&embeddings);
            let weight_hash_before = hash_i8_slice(&output_weights);

            let mut grad_features_q15 = [0_i16; BYTE_EMBED_D_MODEL];
            let mut scaled_grad_output = [0_i32; BYTE_VOCAB];
            linear_backward_input_i16_i8_i16_per_channel_checked(
                &grad_output_q15,
                LinearBackwardInputI16I8Params {
                    weights: &output_weights,
                    forward_scales: &BYTE_EMBED_LOGIT_SCALES,
                    grad_input_scales: &BYTE_EMBED_GRAD_FEATURE_SCALES,
                    input_dim: BYTE_EMBED_D_MODEL,
                    output_dim: BYTE_VOCAB,
                },
                LinearBackwardInputWorkspace {
                    scaled_grad_output: &mut scaled_grad_output,
                },
                &mut grad_features_q15,
            )
            .ok_or(TrainError::CoreRejected("byte_embed_backward_features"))?;

            let mut head_scaled_grad_output = [0_i32; BYTE_VOCAB];
            let head_update = linear_backward_weight_update_i8_checked(
                &features,
                &grad_output_q15,
                &mut output_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &BYTE_EMBED_LOGIT_SCALES,
                    input_dim: BYTE_EMBED_D_MODEL,
                    output_dim: BYTE_VOCAB,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.head_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut head_scaled_grad_output,
                },
            )
            .ok_or(TrainError::CoreRejected("byte_embed_output_head_update"))?;

            let embedding_update = apply_byte_embedding_update(
                &mut embeddings,
                &tokens[window_start..window_start + config.seq_len],
                &grad_features_q15[1..],
                config.learning_rate,
                config.embedding_learning_rate_shift,
                seq_shift,
            )?;

            updates += 1;
            gradient_saturation_count += head_update.gradient_saturation_count;
            embedding_saturation_count += embedding_update.gradient_saturation_count;
            zero_delta_count += head_update.zero_delta_count;
            embedding_zero_delta_count += embedding_update.zero_delta_count;
            weight_delta_l1 = weight_delta_l1.saturating_add(head_update.weight_delta_l1);
            embedding_delta_l1 =
                embedding_delta_l1.saturating_add(embedding_update.weight_delta_l1);

            let after_features =
                byte_embed_features_q15(&embeddings, tokens, window_start, config.seq_len)?;
            let after_row = byte_embed_softmax_row_for(&output_weights, &after_features)?;
            let predicted_token_after = byte_argmax_i32(&after_row.logits_q8);
            let embedding_hash_after = hash_i16_slice(&embeddings);
            let weight_hash_after = hash_i8_slice(&output_weights);

            steps.push(ByteEmbedSoftmaxTrainingStepTrace {
                update_index: updates,
                epoch,
                window_index,
                window_start,
                first_token: tokens[window_start],
                last_token: tokens[window_start + config.seq_len - 1],
                target_token,
                predicted_token_before,
                predicted_token_after,
                target_probability_before_q15: row.probabilities_q15[usize::from(target_token)],
                target_probability_after_q15: after_row.probabilities_q15
                    [usize::from(target_token)],
                embedding_hash_before,
                embedding_hash_after,
                weight_hash_before,
                weight_hash_after,
                gradient_saturation_count: head_update.gradient_saturation_count,
                embedding_saturation_count: embedding_update.gradient_saturation_count,
                zero_delta_count: head_update.zero_delta_count,
                embedding_zero_delta_count: embedding_update.zero_delta_count,
                weight_delta_l1: head_update.weight_delta_l1,
                embedding_delta_l1: embedding_update.weight_delta_l1,
            });
        }
    }

    let final_total_error = byte_embed_total_error(
        tokens,
        &starts,
        &embeddings,
        &output_weights,
        config.seq_len,
    )?;
    let final_probability_error_q15 = byte_embed_total_probability_error_q15(
        tokens,
        &starts,
        &embeddings,
        &output_weights,
        config.seq_len,
    )?;
    let final_mistakes = final_total_error;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = hash_byte_embed_logits(
        tokens,
        &starts,
        &embeddings,
        &output_weights,
        config.seq_len,
    )?;
    let model = ByteEmbedSoftmaxModel {
        seq_len: config.seq_len,
        embeddings,
        output_weights,
    };

    let trace = ByteEmbedSoftmaxTrainingTrace {
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        initial_embedding_hash,
        final_embedding_hash: model.embedding_hash(),
        initial_weight_hash,
        final_weight_hash: model.output_weight_hash(),
        initial_total_error,
        final_total_error,
        initial_probability_error_q15,
        final_probability_error_q15,
        initial_mistakes,
        final_mistakes,
        gradient_saturation_count,
        embedding_saturation_count,
        zero_delta_count,
        embedding_zero_delta_count,
        weight_delta_l1,
        embedding_delta_l1,
        final_accuracy_per_mille,
        final_logits_hash,
        steps,
    };

    Ok(ByteEmbedSoftmaxTrainingRun { trace, model })
}

pub fn run_mini_transformer_mlp_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingTrace, TrainError> {
    Ok(run_mini_transformer_mlp_training_with_model(tokens, config)?.trace)
}

pub fn run_mini_transformer_mlp_training_with_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.learning_rate <= 0
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let starts = byte_window_starts(
        tokens.len(),
        config.seq_len,
        config.stride,
        config.max_windows,
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    let initial_model_hash = model.model_hash();
    let initial_output_head_hash = model.output_head_hash();
    let initial_mlp_hash = model.mlp_hash();
    let initial_attention_hash = model.attention_hash();
    let initial_attention_q_hash = model.attention_q_hash();
    let initial_attention_k_hash = model.attention_k_hash();
    let initial_attention_v_hash = model.attention_v_hash();
    let initial_attention_o_hash = model.attention_o_hash();
    let initial_total_error =
        mini_transformer_total_error(tokens, &starts, &model, config.seq_len)?;
    let initial_probability_error_q15 =
        mini_transformer_total_probability_error_q15(tokens, &starts, &model, config.seq_len)?;
    let initial_mistakes = initial_total_error;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut output_head_saturation_count = 0_usize;
    let mut mlp_saturation_count = 0_usize;
    let mut attention_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;
    let mut output_head_zero_delta_count = 0_usize;
    let mut mlp_zero_delta_count = 0_usize;
    let mut attention_zero_delta_count = 0_usize;
    let mut output_head_delta_l1 = 0_u64;
    let mut mlp_delta_l1 = 0_u64;
    let mut attention_delta_l1 = 0_u64;
    let mut attention_q_delta_l1 = 0_u64;
    let mut attention_k_delta_l1 = 0_u64;
    let mut attention_v_delta_l1 = 0_u64;
    let mut attention_o_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (window_index, &window_start) in starts.iter().enumerate() {
            examined_windows += 1;
            let target_token = tokens[window_start + config.seq_len];
            let cache_before = mini_transformer_forward_for(
                &model,
                &tokens[window_start..window_start + config.seq_len],
            )?;
            let predicted_token_before = byte_argmax_i32(&cache_before.logits_q8);
            let gradient_q15 =
                byte_softmax_gradient_q15(&cache_before.probabilities_q15, target_token);
            let grad_output_q15 = byte_gradient_i32_to_i16(&gradient_q15);
            let output_head_hash_before = model.output_head_hash();
            let mlp_hash_before = model.mlp_hash();
            let attention_hash_before = model.attention_hash();

            let mut grad_last_features = [0_i16; MINI_TRANSFORMER_D_MODEL];
            let mut output_scaled_grad = [0_i32; BYTE_VOCAB];
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
                    scaled_grad_output: &mut output_scaled_grad,
                },
                &mut grad_last_features,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_output_head_backward_input",
            ))?;

            let last_start = (config.seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
            let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
            let output_update = linear_backward_weight_update_i8_checked(
                &cache_before.block_output[last_start..last_end],
                &grad_output_q15,
                &mut model.output_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: BYTE_VOCAB,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.output_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut output_scaled_grad,
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_output_head_update",
            ))?;

            let mut grad_mlp_output = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
            grad_mlp_output[last_start..last_end].copy_from_slice(&grad_last_features);
            let mut grad_mlp_input = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
            let mut mlp_input_scaled_grad =
                vec![0_i32; MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM)];
            let mut mlp_input_grad_gated =
                vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mut mlp_input_grad_up = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mut mlp_input_grad_gate = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mut mlp_input_grad_up_input =
                vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
            let mut mlp_input_grad_gate_input =
                vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
            let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
                &grad_mlp_output,
                mini_transformer_mlp_params(
                    &model.up_weights,
                    &model.gate_weights,
                    &model.down_weights,
                    config.seq_len,
                ),
                &cache_before.mlp_up,
                &cache_before.mlp_gate,
                GatedMlpBackwardScales {
                    down_to_hidden: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    up_to_input: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    gate_to_input: &MINI_TRANSFORMER_D_MODEL_SCALES,
                },
                GatedMlpBackwardWorkspace {
                    scaled_grad_output: &mut mlp_input_scaled_grad,
                    grad_gated: &mut mlp_input_grad_gated,
                    grad_up: &mut mlp_input_grad_up,
                    grad_gate: &mut mlp_input_grad_gate,
                    grad_up_input: &mut mlp_input_grad_up_input,
                    grad_gate_input: &mut mlp_input_grad_gate_input,
                },
                &mut grad_mlp_input,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_mlp_backward_input",
            ))?;

            let mut grad_attention_output = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
            let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                &grad_mlp_output,
                &grad_mlp_input,
                &mut grad_attention_output,
            )?;

            let mut mlp_scaled_grad =
                vec![0_i32; MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM)];
            let mut grad_gated = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mut grad_up = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mut grad_gate = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
            let mlp_update = gated_mlp_backward_weight_update_i8_checked(
                &cache_before.attention_residual,
                &grad_mlp_output,
                &cache_before.mlp_up,
                &cache_before.mlp_gate,
                &cache_before.mlp_gated,
                &mut model.up_weights,
                &mut model.gate_weights,
                &mut model.down_weights,
                GatedMlpWeightUpdateParams {
                    up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    seq_len: config.seq_len,
                    d_model: MINI_TRANSFORMER_D_MODEL,
                    hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.mlp_learning_rate_shift,
                },
                GatedMlpWeightUpdateWorkspace {
                    scaled_grad_output: &mut mlp_scaled_grad,
                    grad_gated: &mut grad_gated,
                    grad_up: &mut grad_up,
                    grad_gate: &mut grad_gate,
                },
            )
            .ok_or(TrainError::CoreRejected("mini_transformer_mlp_update"))?;

            let attention_update = mini_transformer_attention_update_i8_checked(
                &cache_before,
                &grad_attention_output,
                &mut model,
                config,
            )?;

            let cache_after = mini_transformer_forward_for(
                &model,
                &tokens[window_start..window_start + config.seq_len],
            )?;
            let predicted_token_after = byte_argmax_i32(&cache_after.logits_q8);
            let output_head_hash_after = model.output_head_hash();
            let mlp_hash_after = model.mlp_hash();
            let attention_hash_after = model.attention_hash();

            updates += 1;
            output_head_saturation_count += output_update.gradient_saturation_count;
            output_head_zero_delta_count += output_update.zero_delta_count;
            output_head_delta_l1 =
                output_head_delta_l1.saturating_add(output_update.weight_delta_l1);
            mlp_saturation_count += mlp_input_saturation_count;
            mlp_saturation_count += mlp_update.gradient_saturation_count().unwrap_or(usize::MAX);
            mlp_zero_delta_count += mlp_update.zero_delta_count().unwrap_or(usize::MAX);
            mlp_delta_l1 = mlp_delta_l1.saturating_add(mlp_update.weight_delta_l1().unwrap_or(0));
            attention_saturation_count += attention_update.gradient_saturation_count;
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
            residual_saturation_count += cache_before.residual_saturation_count;
            residual_saturation_count += cache_after.residual_saturation_count;

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
                output_head_saturation_count: output_update.gradient_saturation_count,
                mlp_saturation_count: mlp_input_saturation_count
                    + mlp_update.gradient_saturation_count().unwrap_or(usize::MAX),
                attention_saturation_count: attention_update.gradient_saturation_count,
                residual_saturation_count: cache_before.residual_saturation_count
                    + cache_after.residual_saturation_count
                    + gradient_residual_saturation_count,
                output_head_zero_delta_count: output_update.zero_delta_count,
                mlp_zero_delta_count: mlp_update.zero_delta_count().unwrap_or(usize::MAX),
                attention_zero_delta_count: attention_update.zero_delta_count,
                output_head_delta_l1: output_update.weight_delta_l1,
                mlp_delta_l1: mlp_update.weight_delta_l1().unwrap_or(0),
                attention_delta_l1: attention_update.weight_delta_l1,
                attention_q_delta_l1: attention_update.q.weight_delta_l1,
                attention_k_delta_l1: attention_update.k.weight_delta_l1,
                attention_v_delta_l1: attention_update.v.weight_delta_l1,
                attention_o_delta_l1: attention_update.o.weight_delta_l1,
            });
        }
    }

    let final_total_error = mini_transformer_total_error(tokens, &starts, &model, config.seq_len)?;
    let final_probability_error_q15 =
        mini_transformer_total_probability_error_q15(tokens, &starts, &model, config.seq_len)?;
    let final_mistakes = final_total_error;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = hash_mini_transformer_logits(tokens, &starts, &model, config.seq_len)?;

    let trace = MiniTransformerMlpTrainingTrace {
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        initial_model_hash,
        final_model_hash: model.model_hash(),
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
        attention_saturation_count,
        residual_saturation_count,
        output_head_zero_delta_count,
        mlp_zero_delta_count,
        attention_zero_delta_count,
        output_head_delta_l1,
        mlp_delta_l1,
        attention_delta_l1,
        attention_q_delta_l1,
        attention_k_delta_l1,
        attention_v_delta_l1,
        attention_o_delta_l1,
        final_accuracy_per_mille,
        final_logits_hash,
        steps,
    };

    Ok(MiniTransformerMlpTrainingRun { trace, model })
}

impl TrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", TASK);
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", D_MODEL);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "output_head_i8");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "sign_perceptron");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i8_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(&mut out, "learning_rate_shift", 0);
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "samples", self.samples);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "examined_samples",
            self.config.epochs * self.samples,
        );
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(&mut out, "initial_total_error", self.initial_total_error);
        comma(&mut out);
        push_usize_field(&mut out, "final_total_error", self.final_total_error);
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
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.gradient_saturation_count,
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl SoftmaxTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", SOFTMAX_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", SOFTMAX_TASK);
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", D_MODEL);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "output_head_i8");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "logit_scale", "q8");
        comma(&mut out);
        push_string_field(&mut out, "probability_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "learning_rate_shift",
            usize::from(self.config.learning_rate_shift),
        );
        comma(&mut out);
        push_string_field(&mut out, "gradient", "prob_q15_minus_target_q15");
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "samples", self.samples);
        comma(&mut out);
        push_usize_field(&mut out, "examined_samples", self.examined_samples);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        comma(&mut out);
        push_string_field(
            &mut out,
            "stop_reason",
            if self.final_total_error == 0 {
                "zero_classification_error"
            } else {
                "epoch_limit"
            },
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
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.gradient_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "zero_delta_count", self.zero_delta_count);
        comma(&mut out);
        push_u64_field(&mut out, "weight_delta_l1", self.weight_delta_l1);
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_softmax_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &SOFTMAX_KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl LinearBackwardTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", LINEAR_BACKWARD_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", LINEAR_BACKWARD_TASK);
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "input_dim", LINEAR_BACKWARD_INPUT_DIM);
        comma(&mut out);
        push_usize_field(&mut out, "output_dim", LINEAR_BACKWARD_OUTPUT_DIM);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "linear_i16_i8_i16");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "outer_product_sgd_checked");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "grad_output_scale", "q15_prescaled_per_channel");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_string_field(&mut out, "intermediate", "i64_outer_product");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "learning_rate_shift",
            usize::from(self.config.learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"forward\":{");
        push_i16_array_field(&mut out, "input_q15", &self.input_q15);
        comma(&mut out);
        push_fixed_scales_field(&mut out, "forward_scales", &LINEAR_BACKWARD_FORWARD_SCALES);
        comma(&mut out);
        push_i16_array_field(&mut out, "output_before_q15", &self.output_before_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "output_after_q15", &self.output_after_q15);
        out.push('}');
        comma(&mut out);
        out.push_str("\"backward\":{");
        push_i16_array_field(&mut out, "grad_output_q15", &self.grad_output_q15);
        comma(&mut out);
        push_i32_array_field(
            &mut out,
            "scaled_grad_output_i32",
            &self.scaled_grad_output_i32,
        );
        comma(&mut out);
        push_fixed_scales_field(
            &mut out,
            "grad_input_scales",
            &LINEAR_BACKWARD_GRAD_INPUT_SCALES,
        );
        comma(&mut out);
        push_i16_array_field(&mut out, "grad_input_q15", &self.grad_input_q15);
        out.push('}');
        comma(&mut out);
        out.push_str("\"weights\":{");
        push_i8_array_field(&mut out, "before_i8", &self.weights_before_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "after_i8", &self.weights_after_i8);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.update_stats.gradient_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "zero_delta_count",
            self.update_stats.zero_delta_count,
        );
        comma(&mut out);
        push_u64_field(
            &mut out,
            "weight_delta_l1",
            self.update_stats.weight_delta_l1,
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash_before", self.output_hash_before);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash_after", self.output_hash_after);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &LINEAR_BACKWARD_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl GatedMlpBackwardTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", GATED_MLP_BACKWARD_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", GATED_MLP_BACKWARD_TASK);
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "seq_len", GATED_MLP_BACKWARD_SEQ_LEN);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", GATED_MLP_BACKWARD_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", GATED_MLP_BACKWARD_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "gated_mlp_i8_matrices");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "outer_product_sgd_checked");
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
            "learning_rate_shift",
            usize::from(self.config.learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"forward\":{");
        push_i16_array_field(&mut out, "input_q15", &self.input_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "up_q15", &self.forward_up_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "gate_q15", &self.forward_gate_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "gated_q15", &self.forward_gated_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "output_before_q15", &self.output_before_q15);
        comma(&mut out);
        push_i16_array_field(&mut out, "output_after_q15", &self.output_after_q15);
        out.push('}');
        comma(&mut out);
        out.push_str("\"backward\":{");
        push_i16_array_field(&mut out, "grad_output_q15", &self.grad_output_q15);
        out.push('}');
        comma(&mut out);
        out.push_str("\"weights\":{");
        push_i8_array_field(&mut out, "up_before_i8", &self.up_before_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "up_after_i8", &self.up_after_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "gate_before_i8", &self.gate_before_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "gate_after_i8", &self.gate_after_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "down_before_i8", &self.down_before_i8);
        comma(&mut out);
        push_i8_array_field(&mut out, "down_after_i8", &self.down_after_i8);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.update_stats
                .gradient_saturation_count()
                .unwrap_or(usize::MAX),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "zero_delta_count",
            self.update_stats.zero_delta_count().unwrap_or(usize::MAX),
        );
        comma(&mut out);
        push_u64_field(
            &mut out,
            "weight_delta_l1",
            self.update_stats.weight_delta_l1().unwrap_or(u64::MAX),
        );
        comma(&mut out);
        push_linear_weight_update_stats_field(&mut out, "down", self.update_stats.down);
        comma(&mut out);
        push_linear_weight_update_stats_field(&mut out, "up", self.update_stats.up);
        comma(&mut out);
        push_linear_weight_update_stats_field(&mut out, "gate", self.update_stats.gate);
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash_before", self.output_hash_before);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash_after", self.output_hash_after);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &GATED_MLP_BACKWARD_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl ByteSoftmaxTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BYTE_SOFTMAX_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", BYTE_SOFTMAX_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
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
        push_usize_field(&mut out, "d_model", BYTE_D_MODEL);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "byte_output_head_i8");
        comma(&mut out);
        push_string_field(&mut out, "features", "bias_plus_last_byte_one_hot_q15");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "logit_scale", "q8");
        comma(&mut out);
        push_string_field(&mut out, "probability_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "learning_rate_shift",
            usize::from(self.config.learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
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
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.gradient_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "zero_delta_count", self.zero_delta_count);
        comma(&mut out);
        push_u64_field(&mut out, "weight_delta_l1", self.weight_delta_l1);
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_byte_softmax_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &BYTE_SOFTMAX_KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl ByteEmbedSoftmaxTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BYTE_EMBED_SOFTMAX_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", BYTE_EMBED_SOFTMAX_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
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
        push_usize_field(&mut out, "embedding_dim", BYTE_EMBED_DIM);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", BYTE_EMBED_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.config.seq_len);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trained_component",
            "byte_embedding_i16_plus_output_head_i8",
        );
        comma(&mut out);
        push_string_field(&mut out, "features", "bias_plus_mean_byte_embedding_q15");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "embedding_dtype", "i16");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "head_learning_rate_shift",
            usize::from(self.config.head_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
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
        push_usize_field(
            &mut out,
            "gradient_saturation_count",
            self.gradient_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_saturation_count",
            self.embedding_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "zero_delta_count", self.zero_delta_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_zero_delta_count",
            self.embedding_zero_delta_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "weight_delta_l1", self.weight_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_delta_l1", self.embedding_delta_l1);
        out.push('}');
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_embedding_hash",
            self.initial_embedding_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_embedding_hash", self.final_embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "initial_weight_hash", self.initial_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_weight_hash", self.final_weight_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_byte_embed_softmax_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &BYTE_EMBED_SOFTMAX_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
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
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
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
            "output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8",
        );
        comma(&mut out);
        push_string_field(&mut out, "attention", "updates_q_k_v_o_i8");
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
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
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
            "attention_zero_delta_count",
            self.attention_zero_delta_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "output_head_delta_l1", self.output_head_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_delta_l1", self.mlp_delta_l1);
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
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_model_hash", self.initial_model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_model_hash", self.final_model_hash);
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
}

impl ByteSoftmaxModel {
    pub fn new(weights: Vec<i8>) -> Result<Self, TrainError> {
        if weights.len() != BYTE_VOCAB * BYTE_D_MODEL {
            return Err(TrainError::InvalidModel("wrong byte-softmax weight count"));
        }
        Ok(Self { weights })
    }

    pub fn weight_hash(&self) -> u64 {
        hash_i8_slice(&self.weights)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.weights.len());
        out.extend_from_slice(BYTE_SOFTMAX_MODEL_MAGIC);
        out.extend_from_slice(&(BYTE_VOCAB as u32).to_le_bytes());
        out.extend_from_slice(&(BYTE_D_MODEL as u32).to_le_bytes());
        out.extend_from_slice(&(self.weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.weight_hash().to_le_bytes());
        out.extend(self.weights.iter().map(|&weight| weight as u8));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = BYTE_SOFTMAX_MODEL_MAGIC.len() + 4 + 4 + 8 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        if &bytes[..BYTE_SOFTMAX_MODEL_MAGIC.len()] != BYTE_SOFTMAX_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = BYTE_SOFTMAX_MODEL_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)?;
        let d_model = read_u32_le(bytes, &mut offset)?;
        let weight_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_hash = read_u64_le(bytes, &mut offset)?;

        if vocab as usize != BYTE_VOCAB || d_model as usize != BYTE_D_MODEL {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }
        if weight_count != BYTE_VOCAB * BYTE_D_MODEL {
            return Err(TrainError::InvalidModel("weight count mismatch"));
        }
        if bytes.len() != offset + weight_count {
            return Err(TrainError::InvalidModel("artifact length mismatch"));
        }

        let weights = bytes[offset..]
            .iter()
            .map(|&byte| byte as i8)
            .collect::<Vec<_>>();
        let model = Self::new(weights)?;
        if model.weight_hash() != expected_hash {
            return Err(TrainError::InvalidModel("weight hash mismatch"));
        }
        Ok(model)
    }
}

impl ByteEmbedSoftmaxModel {
    pub fn new(
        seq_len: usize,
        embeddings: Vec<i16>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        if seq_len == 0 || !seq_len.is_power_of_two() {
            return Err(TrainError::InvalidModel("bad context seq_len"));
        }
        if embeddings.len() != BYTE_VOCAB * BYTE_EMBED_DIM {
            return Err(TrainError::InvalidModel("wrong byte embedding count"));
        }
        if output_weights.len() != BYTE_VOCAB * BYTE_EMBED_D_MODEL {
            return Err(TrainError::InvalidModel("wrong byte output weight count"));
        }
        Ok(Self {
            seq_len,
            embeddings,
            output_weights,
        })
    }

    pub fn embedding_hash(&self) -> u64 {
        hash_i16_slice(&self.embeddings)
    }

    pub fn output_weight_hash(&self) -> u64 {
        hash_i8_slice(&self.output_weights)
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.seq_len);
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i8_slice(&self.output_weights);
        hasher.finish()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(52 + self.embeddings.len() * 2 + self.output_weights.len());
        out.extend_from_slice(BYTE_EMBED_SOFTMAX_MODEL_MAGIC);
        out.extend_from_slice(&(BYTE_VOCAB as u32).to_le_bytes());
        out.extend_from_slice(&(BYTE_EMBED_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u32).to_le_bytes());
        out.extend_from_slice(&(self.embeddings.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.output_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.embedding_hash().to_le_bytes());
        out.extend_from_slice(&self.output_weight_hash().to_le_bytes());
        for &embedding in self.embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        out.extend(self.output_weights.iter().map(|&weight| weight as u8));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = BYTE_EMBED_SOFTMAX_MODEL_MAGIC.len() + 4 + 4 + 4 + 8 + 8 + 8 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        if &bytes[..BYTE_EMBED_SOFTMAX_MODEL_MAGIC.len()] != BYTE_EMBED_SOFTMAX_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = BYTE_EMBED_SOFTMAX_MODEL_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)?;
        let embed_dim = read_u32_le(bytes, &mut offset)?;
        let seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let weight_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_embedding_hash = read_u64_le(bytes, &mut offset)?;
        let expected_weight_hash = read_u64_le(bytes, &mut offset)?;

        if vocab as usize != BYTE_VOCAB || embed_dim as usize != BYTE_EMBED_DIM {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }
        if seq_len == 0 || !seq_len.is_power_of_two() {
            return Err(TrainError::InvalidModel("bad context seq_len"));
        }
        if embedding_count != BYTE_VOCAB * BYTE_EMBED_DIM {
            return Err(TrainError::InvalidModel("embedding count mismatch"));
        }
        if weight_count != BYTE_VOCAB * BYTE_EMBED_D_MODEL {
            return Err(TrainError::InvalidModel("weight count mismatch"));
        }

        let embedding_bytes = embedding_count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("embedding length overflow"))?;
        let embedding_end = offset
            .checked_add(embedding_bytes)
            .ok_or(TrainError::InvalidModel("embedding offset overflow"))?;
        let weight_end = embedding_end
            .checked_add(weight_count)
            .ok_or(TrainError::InvalidModel("weight offset overflow"))?;
        if bytes.len() != weight_end {
            return Err(TrainError::InvalidModel("artifact length mismatch"));
        }

        let mut embeddings = Vec::with_capacity(embedding_count);
        for chunk in bytes[offset..embedding_end].chunks_exact(2) {
            embeddings.push(i16::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TrainError::InvalidModel("bad embedding"))?,
            ));
        }
        let output_weights = bytes[embedding_end..weight_end]
            .iter()
            .map(|&byte| byte as i8)
            .collect::<Vec<_>>();
        let model = Self::new(seq_len, embeddings, output_weights)?;
        if model.embedding_hash() != expected_embedding_hash {
            return Err(TrainError::InvalidModel("embedding hash mismatch"));
        }
        if model.output_weight_hash() != expected_weight_hash {
            return Err(TrainError::InvalidModel("weight hash mismatch"));
        }
        Ok(model)
    }
}

impl MiniTransformerMlpModel {
    pub fn new_initial() -> Self {
        Self::new_initial_with_seq_len(DEFAULT_MINI_TRANSFORMER_SEQ_LEN)
    }

    pub fn new_initial_with_seq_len(context_seq_len: usize) -> Self {
        Self {
            context_seq_len,
            embeddings: initial_mini_transformer_embeddings(),
            q_weights: identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
            k_weights: identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
            v_weights: identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
            o_weights: identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
            up_weights: initial_mini_transformer_mlp_up_or_gate_weights(),
            gate_weights: initial_mini_transformer_mlp_up_or_gate_weights(),
            down_weights: initial_mini_transformer_mlp_down_weights(),
            output_weights: initial_mini_transformer_output_weights(),
        }
    }

    pub fn new(
        context_seq_len: usize,
        embeddings: Vec<i16>,
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
        let attention_weight_count = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL;
        if q_weights.len() != attention_weight_count
            || k_weights.len() != attention_weight_count
            || v_weights.len() != attention_weight_count
            || o_weights.len() != attention_weight_count
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer attention weight count",
            ));
        }
        let up_or_gate_count = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM;
        if up_weights.len() != up_or_gate_count || gate_weights.len() != up_or_gate_count {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer up/gate weight count",
            ));
        }
        if down_weights.len() != MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL {
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

    pub fn embedding_hash(&self) -> u64 {
        hash_i16_slice(&self.embeddings)
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

    pub fn to_bytes(&self) -> Vec<u8> {
        let embedding_bytes = self.embeddings.len() * 2;
        let weight_bytes = self.q_weights.len()
            + self.k_weights.len()
            + self.v_weights.len()
            + self.o_weights.len()
            + self.up_weights.len()
            + self.gate_weights.len()
            + self.down_weights.len()
            + self.output_weights.len();
        let mut out = Vec::with_capacity(128 + embedding_bytes + weight_bytes);
        out.extend_from_slice(MINI_TRANSFORMER_MODEL_MAGIC);
        out.extend_from_slice(&(BYTE_VOCAB as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_D_MODEL as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_HEADS as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_HIDDEN_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(self.context_seq_len as u32).to_le_bytes());
        out.extend_from_slice(&(self.embeddings.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.q_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.k_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.v_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.o_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.up_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.gate_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.down_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.output_weights.len() as u64).to_le_bytes());
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
        out.extend(self.q_weights.iter().map(|&weight| weight as u8));
        out.extend(self.k_weights.iter().map(|&weight| weight as u8));
        out.extend(self.v_weights.iter().map(|&weight| weight as u8));
        out.extend(self.o_weights.iter().map(|&weight| weight as u8));
        out.extend(self.up_weights.iter().map(|&weight| weight as u8));
        out.extend(self.gate_weights.iter().map(|&weight| weight as u8));
        out.extend(self.down_weights.iter().map(|&weight| weight as u8));
        out.extend(self.output_weights.iter().map(|&weight| weight as u8));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 9 + 8 * 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_MODEL_MAGIC.len()] != MINI_TRANSFORMER_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = MINI_TRANSFORMER_MODEL_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let context_seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
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

        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }

        let expected_attention_count = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL;
        if embedding_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
            || q_count != expected_attention_count
            || k_count != expected_attention_count
            || v_count != expected_attention_count
            || o_count != expected_attention_count
            || up_count != MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM
            || gate_count != MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM
            || down_count != MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL
            || output_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        {
            return Err(TrainError::InvalidModel("tensor count mismatch"));
        }

        let embedding_bytes = embedding_count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("embedding length overflow"))?;
        let weight_bytes = q_count
            .checked_add(k_count)
            .and_then(|value| value.checked_add(v_count))
            .and_then(|value| value.checked_add(o_count))
            .and_then(|value| value.checked_add(up_count))
            .and_then(|value| value.checked_add(gate_count))
            .and_then(|value| value.checked_add(down_count))
            .and_then(|value| value.checked_add(output_count))
            .ok_or(TrainError::InvalidModel("weight length overflow"))?;
        let expected_len = offset
            .checked_add(embedding_bytes)
            .and_then(|value| value.checked_add(weight_bytes))
            .ok_or(TrainError::InvalidModel("artifact length overflow"))?;
        if bytes.len() != expected_len {
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

        let q_weights = read_i8_vec(bytes, &mut offset, q_count)?;
        let k_weights = read_i8_vec(bytes, &mut offset, k_count)?;
        let v_weights = read_i8_vec(bytes, &mut offset, v_count)?;
        let o_weights = read_i8_vec(bytes, &mut offset, o_count)?;
        let up_weights = read_i8_vec(bytes, &mut offset, up_count)?;
        let gate_weights = read_i8_vec(bytes, &mut offset, gate_count)?;
        let down_weights = read_i8_vec(bytes, &mut offset, down_count)?;
        let output_weights = read_i8_vec(bytes, &mut offset, output_count)?;

        let model = Self::new(
            context_seq_len,
            embeddings,
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

impl ByteGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BYTE_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", BYTE_SOFTMAX_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
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
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &BYTE_GENERATION_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl ByteEmbedGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BYTE_EMBED_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", BYTE_EMBED_SOFTMAX_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_weight_hash", self.output_weight_hash);
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
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &BYTE_EMBED_GENERATION_KNOWN_NON_CLAIMS,
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
        push_string_field(&mut out, "tokenizer", BYTE_TOKENIZER_ID);
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
        push_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

pub fn generate_byte_softmax(
    model: &ByteSoftmaxModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<ByteGenerationTrace, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let features = byte_single_token_features_q15(input_token);
        let row = byte_softmax_row_for(&model.weights, &features)?;
        let predicted_token = byte_argmax_i32(&row.logits_q8);
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: row.logits_q8[predicted_index],
            predicted_probability_q15: row.probabilities_q15[predicted_index],
        });
    }

    Ok(ByteGenerationTrace {
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.weight_hash(),
        steps,
    })
}

pub fn generate_byte_embed_softmax(
    model: &ByteEmbedSoftmaxModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<ByteEmbedGenerationTrace, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    byte_embed_seq_shift(model.seq_len)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let features = byte_embed_context_features_q15(&model.embeddings, &context, model.seq_len)?;
        let row = byte_embed_softmax_row_for(&model.output_weights, &features)?;
        let predicted_token = byte_argmax_i32(&row.logits_q8);
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: row.logits_q8[predicted_index],
            predicted_probability_q15: row.probabilities_q15[predicted_index],
        });
    }

    Ok(ByteEmbedGenerationTrace {
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        output_weight_hash: model.output_weight_hash(),
        context_seq_len: model.seq_len,
        steps,
    })
}

pub fn generate_mini_transformer(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    if prompt.is_empty() || model.context_seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let cache = mini_transformer_forward_for(model, &context[context_start..])?;
        let predicted_token = byte_argmax_i32(&cache.logits_q8);
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: cache.logits_q8[predicted_index],
            predicted_probability_q15: cache.probabilities_q15[predicted_index],
        });
    }

    Ok(MiniTransformerGenerationTrace {
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        steps,
    })
}

fn total_error(weights: &[i8]) -> Result<usize, TrainError> {
    count_mistakes(weights)
}

fn total_probability_error_q15(weights: &[i8]) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &(input_id, target_id) in TRAINING_PAIRS.iter() {
        let row = softmax_row_for(weights, input_id)?;
        error = error.saturating_add(sample_probability_error_q15(
            &row.probabilities_q15,
            target_id,
        ));
    }
    Ok(error)
}

fn sample_probability_error_q15(probabilities_q15: &[i16; VOCAB], target_id: usize) -> usize {
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities_q15[target_id])).max(0) as usize;
    for (class_id, &probability) in probabilities_q15.iter().enumerate() {
        if class_id != target_id {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
}

fn count_mistakes(weights: &[i8]) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for &(input_id, target_id) in TRAINING_PAIRS.iter() {
        let logits = logits_for(weights, input_id)?;
        if argmax(&logits) != target_id {
            mistakes += 1;
        }
    }
    Ok(mistakes)
}

fn logits_for(weights: &[i8], input_id: usize) -> Result<[i16; VOCAB], TrainError> {
    let params = LinearI16I8Params {
        weights,
        bias: None,
        scales: &LOGIT_SCALES,
        input_dim: D_MODEL,
        output_dim: VOCAB,
    };
    let mut logits = [0_i16; VOCAB];
    linear_i16_i8_i16_per_channel_checked(&FEATURES_Q15[input_id], params, &mut logits)
        .ok_or(TrainError::CoreRejected("linear_output_head"))?;
    Ok(logits)
}

fn linear_backward_output_for(
    weights: &[i8],
) -> Result<[i16; LINEAR_BACKWARD_OUTPUT_DIM], TrainError> {
    let params = LinearI16I8Params {
        weights,
        bias: None,
        scales: &LINEAR_BACKWARD_FORWARD_SCALES,
        input_dim: LINEAR_BACKWARD_INPUT_DIM,
        output_dim: LINEAR_BACKWARD_OUTPUT_DIM,
    };
    let mut output = [0_i16; LINEAR_BACKWARD_OUTPUT_DIM];
    linear_i16_i8_i16_per_channel_checked(&LINEAR_BACKWARD_INPUT_Q15, params, &mut output)
        .ok_or(TrainError::CoreRejected("linear_backward_forward_replay"))?;
    Ok(output)
}

fn gated_mlp_backward_forward_for(
    up_weights: &[i8],
    gate_weights: &[i8],
    down_weights: &[i8],
) -> Result<
    (
        [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
        [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
        [i16; GATED_MLP_BACKWARD_HIDDEN_DIM],
        [i16; GATED_MLP_BACKWARD_D_MODEL],
    ),
    TrainError,
> {
    let params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: up_weights,
            bias: None,
            scales: &GATED_MLP_BACKWARD_HIDDEN_SCALES,
            input_dim: GATED_MLP_BACKWARD_D_MODEL,
            output_dim: GATED_MLP_BACKWARD_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: gate_weights,
            bias: None,
            scales: &GATED_MLP_BACKWARD_HIDDEN_SCALES,
            input_dim: GATED_MLP_BACKWARD_D_MODEL,
            output_dim: GATED_MLP_BACKWARD_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: down_weights,
            bias: None,
            scales: &GATED_MLP_BACKWARD_D_MODEL_SCALES,
            input_dim: GATED_MLP_BACKWARD_HIDDEN_DIM,
            output_dim: GATED_MLP_BACKWARD_D_MODEL,
        },
        seq_len: GATED_MLP_BACKWARD_SEQ_LEN,
        d_model: GATED_MLP_BACKWARD_D_MODEL,
        hidden_dim: GATED_MLP_BACKWARD_HIDDEN_DIM,
    };
    let mut up = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let mut gate = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let mut gated = [0_i16; GATED_MLP_BACKWARD_HIDDEN_DIM];
    let mut output = [0_i16; GATED_MLP_BACKWARD_D_MODEL];
    gated_mlp_i16_q15_checked(
        &GATED_MLP_BACKWARD_INPUT_Q15,
        params,
        GatedMlpWorkspace {
            up: &mut up,
            gate: &mut gate,
            gated: &mut gated,
        },
        &mut output,
    )
    .ok_or(TrainError::CoreRejected("gated_mlp_forward_replay"))?;
    Ok((up, gate, gated, output))
}

fn byte_window_starts(
    token_count: usize,
    seq_len: usize,
    stride: usize,
    max_windows: Option<usize>,
) -> Vec<usize> {
    let mut starts = Vec::new();
    if seq_len == 0 || stride == 0 {
        return starts;
    }

    let mut start = 0_usize;
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

fn byte_features_q15(tokens: &[u8], window_start: usize, seq_len: usize) -> [i16; BYTE_D_MODEL] {
    byte_single_token_features_q15(tokens[window_start + seq_len - 1])
}

fn byte_single_token_features_q15(token: u8) -> [i16; BYTE_D_MODEL] {
    let mut features = [0_i16; BYTE_D_MODEL];
    features[0] = 4096;
    features[1 + usize::from(token)] = 8192;
    features
}

fn byte_embed_seq_shift(seq_len: usize) -> Result<u8, TrainError> {
    if seq_len == 0 || !seq_len.is_power_of_two() || seq_len.trailing_zeros() > u32::from(u8::MAX) {
        return Err(TrainError::InvalidConfig);
    }
    Ok(seq_len.trailing_zeros() as u8)
}

fn initial_byte_embed_embeddings() -> Vec<i16> {
    let mut embeddings = Vec::with_capacity(BYTE_VOCAB * BYTE_EMBED_DIM);
    for token in 0..BYTE_VOCAB {
        for dim in 0..BYTE_EMBED_DIM {
            let bucket = ((token * 37 + dim * 17 + 11) % 33) as i32 - 16;
            embeddings.push((bucket * 32) as i16);
        }
    }
    embeddings
}

fn initial_byte_embed_output_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(BYTE_VOCAB * BYTE_EMBED_D_MODEL);
    for class_id in 0..BYTE_VOCAB {
        for feature_id in 0..BYTE_EMBED_D_MODEL {
            let value = if feature_id == 0 {
                0
            } else {
                ((class_id * 19 + feature_id * 23 + 7) % 5) as i32 - 2
            };
            weights.push(value as i8);
        }
    }
    weights
}

fn initial_mini_transformer_embeddings() -> Vec<i16> {
    let mut embeddings = Vec::with_capacity(BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL);
    for token in 0..BYTE_VOCAB {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let bucket = ((token * 29 + dim * 13 + 5) % 33) as i32 - 16;
            embeddings.push((bucket * 32) as i16);
        }
    }
    embeddings
}

fn identity_i8_matrix(dim: usize) -> Vec<i8> {
    let mut weights = vec![0_i8; dim * dim];
    for index in 0..dim {
        weights[index * dim + index] = 1;
    }
    weights
}

fn initial_mini_transformer_mlp_up_or_gate_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM);
    for hidden in 0..MINI_TRANSFORMER_HIDDEN_DIM {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let value = ((hidden * 7 + dim * 11 + 3) % 5) as i32 - 2;
            weights.push(value as i8);
        }
    }
    weights
}

fn initial_mini_transformer_mlp_down_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL);
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        for hidden in 0..MINI_TRANSFORMER_HIDDEN_DIM {
            let value = ((dim * 17 + hidden * 5 + 1) % 5) as i32 - 2;
            weights.push(value as i8);
        }
    }
    weights
}

fn initial_mini_transformer_output_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL);
    for class_id in 0..BYTE_VOCAB {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let value = ((class_id * 19 + dim * 23 + 7) % 7) as i32 - 3;
            weights.push(value as i8);
        }
    }
    weights
}

fn byte_embed_features_q15(
    embeddings: &[i16],
    tokens: &[u8],
    window_start: usize,
    seq_len: usize,
) -> Result<[i16; BYTE_EMBED_D_MODEL], TrainError> {
    let window_end = window_start
        .checked_add(seq_len)
        .ok_or(TrainError::InvalidConfig)?;
    let context = tokens
        .get(window_start..window_end)
        .ok_or(TrainError::InvalidConfig)?;
    byte_embed_context_features_q15(embeddings, context, seq_len)
}

fn byte_embed_context_features_q15(
    embeddings: &[i16],
    context: &[u8],
    seq_len: usize,
) -> Result<[i16; BYTE_EMBED_D_MODEL], TrainError> {
    if embeddings.len() != BYTE_VOCAB * BYTE_EMBED_DIM {
        return Err(TrainError::InvalidModel("wrong byte embedding count"));
    }
    let seq_shift = byte_embed_seq_shift(seq_len)?;
    let mut accumulators = [0_i32; BYTE_EMBED_DIM];
    let context_len = context.len().min(seq_len);
    let pad_tokens = seq_len - context_len;

    for _ in 0..pad_tokens {
        add_byte_embedding_to_accumulators(embeddings, 0, &mut accumulators)?;
    }
    for &token in &context[context.len() - context_len..] {
        add_byte_embedding_to_accumulators(embeddings, token, &mut accumulators)?;
    }

    let mut features = [0_i16; BYTE_EMBED_D_MODEL];
    features[0] = 4096;
    for (dim, &acc) in accumulators.iter().enumerate() {
        features[dim + 1] = saturate_i16(round_shift_rhu_i64(i64::from(acc), seq_shift));
    }
    Ok(features)
}

fn add_byte_embedding_to_accumulators(
    embeddings: &[i16],
    token: u8,
    accumulators: &mut [i32; BYTE_EMBED_DIM],
) -> Result<(), TrainError> {
    let row_start = usize::from(token) * BYTE_EMBED_DIM;
    let row = embeddings
        .get(row_start..row_start + BYTE_EMBED_DIM)
        .ok_or(TrainError::InvalidModel("embedding row out of bounds"))?;
    for (acc, &embedding) in accumulators.iter_mut().zip(row.iter()) {
        *acc = acc.saturating_add(i32::from(embedding));
    }
    Ok(())
}

fn mini_transformer_embedding_sequence_q15(
    embeddings: &[i16],
    context: &[u8],
) -> Result<Vec<i16>, TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL || context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut output = Vec::with_capacity(context.len() * MINI_TRANSFORMER_D_MODEL);
    for &token in context {
        let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        let row = embeddings
            .get(row_start..row_start + MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("mini transformer embedding row"))?;
        output.extend_from_slice(row);
    }
    Ok(output)
}

fn mini_transformer_forward_for(
    model: &MiniTransformerMlpModel,
    context: &[u8],
) -> Result<MiniTransformerMlpForwardCache, TrainError> {
    if context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let seq_len = context.len();
    let total = seq_len * MINI_TRANSFORMER_D_MODEL;
    let hidden_total = seq_len * MINI_TRANSFORMER_HIDDEN_DIM;
    let embedding_output = mini_transformer_embedding_sequence_q15(&model.embeddings, context)?;

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
    self_attention_i16_q15_checked(
        &embedding_output,
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
    let attention_probabilities_q15 =
        mini_transformer_attention_probabilities_q15(seq_len, &q, &k)?;

    let mut residual_saturation_count = 0_usize;
    let mut attention_residual = vec![0_i16; total];
    residual_saturation_count += add_i16_residual_rows_checked(
        &embedding_output,
        &attention_output,
        &mut attention_residual,
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
        seq_len,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    let mut mlp_up = vec![0_i16; hidden_total];
    let mut mlp_gate = vec![0_i16; hidden_total];
    let mut mlp_gated = vec![0_i16; hidden_total];
    let mut mlp_output = vec![0_i16; total];
    gated_mlp_i16_q15_checked(
        &attention_residual,
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

    let last_start = (seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
    let row = mini_transformer_output_row_for(
        &model.output_weights,
        &block_output[last_start..last_end],
    )?;

    Ok(MiniTransformerMlpForwardCache {
        embedding_output,
        attention_q: q,
        attention_k: k,
        attention_v: v,
        attention_context,
        attention_probabilities_q15,
        attention_output,
        attention_residual,
        mlp_up,
        mlp_gate,
        mlp_gated,
        mlp_output,
        block_output,
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
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

fn mini_transformer_attention_probabilities_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
) -> Result<Vec<i16>, TrainError> {
    if seq_len == 0 || MINI_TRANSFORMER_HEADS != 1 {
        return Err(TrainError::InvalidConfig);
    }

    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if q.len() != total || k.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut probabilities = vec![0_i16; seq_len * seq_len];
    let mut logits = vec![0_i32; seq_len];
    for query_index in 0..seq_len {
        let query_start = query_index * MINI_TRANSFORMER_D_MODEL;
        let query_end = query_start + MINI_TRANSFORMER_D_MODEL;
        for key_index in 0..seq_len {
            if key_index > query_index {
                logits[key_index] = MASKED_LOGIT;
                continue;
            }

            let key_start = key_index * MINI_TRANSFORMER_D_MODEL;
            let key_end = key_start + MINI_TRANSFORMER_D_MODEL;
            logits[key_index] = attention_dot_q_k_i16_i32_checked(
                &q[query_start..query_end],
                &k[key_start..key_end],
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_probability_logits",
            ))?;
        }

        let prob_start = query_index * seq_len;
        let prob_end = prob_start + seq_len;
        base2_softmax_i32_q15(&logits, &mut probabilities[prob_start..prob_end]).ok_or(
            TrainError::CoreRejected("mini_transformer_attention_probability_softmax"),
        )?;
    }

    Ok(probabilities)
}

fn mini_transformer_attention_update_i8_checked(
    cache: &MiniTransformerMlpForwardCache,
    grad_attention_output: &[i16],
    model: &mut MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let seq_len = cache.embedding_output.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || cache.embedding_output.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
        || cache.attention_probabilities_q15.len()
            != seq_len
                .checked_mul(seq_len)
                .ok_or(TrainError::InvalidConfig)?
        || grad_attention_output.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_context = vec![0_i16; total];
    let mut scaled_grad = [0_i32; MINI_TRANSFORMER_D_MODEL];

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_attention_output[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.o_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
            &mut grad_context[row_start..row_end],
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_o_backward_input",
        ))?;
    }

    let grad_v = mini_transformer_attention_v_gradient_q15(
        seq_len,
        &cache.attention_probabilities_q15,
        &grad_context,
    )?;
    let grad_probabilities = mini_transformer_attention_probability_gradient_q15(
        seq_len,
        &cache.attention_v,
        &grad_context,
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

    let mut total_stats = empty_linear_weight_update_stats();
    let mut q_total = empty_linear_weight_update_stats();
    let mut k_total = empty_linear_weight_update_stats();
    let mut v_total = empty_linear_weight_update_stats();
    let mut o_total = empty_linear_weight_update_stats();
    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        let q_stats = linear_backward_weight_update_i8_checked(
            &cache.embedding_output[row_start..row_end],
            &grad_q[row_start..row_end],
            &mut model.q_weights,
            LinearBackwardWeightUpdateI8Params {
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
                learning_rate: config.learning_rate,
                learning_rate_shift: config.attention_qk_learning_rate_shift,
            },
            LinearBackwardWeightUpdateWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_q_update",
        ))?;
        add_linear_weight_update_stats_checked(&mut total_stats, q_stats)?;
        add_linear_weight_update_stats_checked(&mut q_total, q_stats)?;

        let k_stats = linear_backward_weight_update_i8_checked(
            &cache.embedding_output[row_start..row_end],
            &grad_k[row_start..row_end],
            &mut model.k_weights,
            LinearBackwardWeightUpdateI8Params {
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
                learning_rate: config.learning_rate,
                learning_rate_shift: config.attention_qk_learning_rate_shift,
            },
            LinearBackwardWeightUpdateWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_k_update",
        ))?;
        add_linear_weight_update_stats_checked(&mut total_stats, k_stats)?;
        add_linear_weight_update_stats_checked(&mut k_total, k_stats)?;

        let v_stats = linear_backward_weight_update_i8_checked(
            &cache.embedding_output[row_start..row_end],
            &grad_v[row_start..row_end],
            &mut model.v_weights,
            LinearBackwardWeightUpdateI8Params {
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
                learning_rate: config.learning_rate,
                learning_rate_shift: config.attention_learning_rate_shift,
            },
            LinearBackwardWeightUpdateWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_v_update",
        ))?;
        add_linear_weight_update_stats_checked(&mut total_stats, v_stats)?;
        add_linear_weight_update_stats_checked(&mut v_total, v_stats)?;

        let o_stats = linear_backward_weight_update_i8_checked(
            &cache.attention_context[row_start..row_end],
            &grad_attention_output[row_start..row_end],
            &mut model.o_weights,
            LinearBackwardWeightUpdateI8Params {
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
                learning_rate: config.learning_rate,
                learning_rate_shift: config.attention_learning_rate_shift,
            },
            LinearBackwardWeightUpdateWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_o_update",
        ))?;
        add_linear_weight_update_stats_checked(&mut total_stats, o_stats)?;
        add_linear_weight_update_stats_checked(&mut o_total, o_stats)?;
    }

    Ok(MiniTransformerAttentionWeightUpdateStats {
        q: q_total,
        k: k_total,
        v: v_total,
        o: o_total,
        gradient_saturation_count: total_stats.gradient_saturation_count,
        zero_delta_count: total_stats.zero_delta_count,
        weight_delta_l1: total_stats.weight_delta_l1,
    })
}

fn mini_transformer_attention_probability_gradient_q15(
    seq_len: usize,
    values: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0 || values.len() != total || grad_context.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_probabilities = vec![0_i16; seq_len * seq_len];
    for query_index in 0..seq_len {
        for key_index in 0..seq_len {
            if key_index > query_index {
                continue;
            }

            let mut acc = 0_i64;
            for dim in 0..MINI_TRANSFORMER_D_MODEL {
                let grad = grad_context[query_index * MINI_TRANSFORMER_D_MODEL + dim];
                let value = values[key_index * MINI_TRANSFORMER_D_MODEL + dim];
                let product = i64::from(grad).checked_mul(i64::from(value)).ok_or(
                    TrainError::CoreRejected("mini_transformer_attention_probability_gradient"),
                )?;
                acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                    "mini_transformer_attention_probability_gradient_accumulate",
                ))?;
            }

            grad_probabilities[query_index * seq_len + key_index] =
                saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
        }
    }

    Ok(grad_probabilities)
}

fn mini_transformer_attention_logit_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_probabilities_q15: &[i16],
) -> Result<Vec<i16>, TrainError> {
    if seq_len == 0
        || probabilities_q15.len()
            != seq_len
                .checked_mul(seq_len)
                .ok_or(TrainError::InvalidConfig)?
        || grad_probabilities_q15.len()
            != seq_len
                .checked_mul(seq_len)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_logits = vec![0_i16; seq_len * seq_len];
    for query_index in 0..seq_len {
        let row_start = query_index * seq_len;
        let row_end = row_start + seq_len;
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
            weighted_grad = weighted_grad
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

    Ok(grad_logits)
}

fn mini_transformer_attention_q_k_gradients_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    grad_logits_q15: &[i16],
) -> Result<(Vec<i16>, Vec<i16>), TrainError> {
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || q.len() != total
        || k.len() != total
        || grad_logits_q15.len()
            != seq_len
                .checked_mul(seq_len)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    let sqrt_shift = sqrt_power_of_four_shift(MINI_TRANSFORMER_D_MODEL).ok_or(
        TrainError::CoreRejected("mini_transformer_attention_qk_sqrt_shift"),
    )?;
    let mut grad_q = vec![0_i16; total];
    let mut grad_k = vec![0_i16; total];

    for query_index in 0..seq_len {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let mut acc = 0_i64;
            for key_index in 0..=query_index {
                let grad_logit = grad_logits_q15[query_index * seq_len + key_index];
                if grad_logit == 0 {
                    continue;
                }
                let key = k[key_index * MINI_TRANSFORMER_D_MODEL + dim];
                let product = i64::from(grad_logit).checked_mul(i64::from(key)).ok_or(
                    TrainError::CoreRejected("mini_transformer_attention_q_gradient_product"),
                )?;
                acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                    "mini_transformer_attention_q_gradient_accumulate",
                ))?;
            }
            grad_q[query_index * MINI_TRANSFORMER_D_MODEL + dim] =
                saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
        }
    }

    for key_index in 0..seq_len {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let mut acc = 0_i64;
            for query_index in key_index..seq_len {
                let grad_logit = grad_logits_q15[query_index * seq_len + key_index];
                if grad_logit == 0 {
                    continue;
                }
                let query = q[query_index * MINI_TRANSFORMER_D_MODEL + dim];
                let product = i64::from(grad_logit).checked_mul(i64::from(query)).ok_or(
                    TrainError::CoreRejected("mini_transformer_attention_k_gradient_product"),
                )?;
                acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                    "mini_transformer_attention_k_gradient_accumulate",
                ))?;
            }
            grad_k[key_index * MINI_TRANSFORMER_D_MODEL + dim] =
                saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
        }
    }

    Ok((grad_q, grad_k))
}

fn mini_transformer_attention_v_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || probabilities_q15.len()
            != seq_len
                .checked_mul(seq_len)
                .ok_or(TrainError::InvalidConfig)?
        || grad_context.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_v = vec![0_i16; total];
    for key_index in 0..seq_len {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let mut acc = 0_i64;
            for query_index in 0..seq_len {
                let probability = probabilities_q15[query_index * seq_len + key_index];
                if probability < 0 {
                    return Err(TrainError::CoreRejected(
                        "mini_transformer_attention_v_negative_probability",
                    ));
                }
                if probability == 0 {
                    continue;
                }

                let grad = grad_context[query_index * MINI_TRANSFORMER_D_MODEL + dim];
                let product = i64::from(probability).checked_mul(i64::from(grad)).ok_or(
                    TrainError::CoreRejected("mini_transformer_attention_v_gradient_product"),
                )?;
                acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                    "mini_transformer_attention_v_gradient_accumulate",
                ))?;
            }

            grad_v[key_index * MINI_TRANSFORMER_D_MODEL + dim] =
                saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
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

fn mini_transformer_output_row_for(
    output_weights: &[i8],
    features: &[i16],
) -> Result<ByteSoftmaxRow, TrainError> {
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

    Ok(ByteSoftmaxRow {
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

fn byte_total_error(
    tokens: &[u8],
    starts: &[usize],
    weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for &start in starts {
        let features = byte_features_q15(tokens, start, seq_len);
        let row = byte_softmax_row_for(weights, &features)?;
        if byte_argmax_i32(&row.logits_q8) != tokens[start + seq_len] {
            mistakes += 1;
        }
    }
    Ok(mistakes)
}

fn byte_total_probability_error_q15(
    tokens: &[u8],
    starts: &[usize],
    weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &start in starts {
        let features = byte_features_q15(tokens, start, seq_len);
        let row = byte_softmax_row_for(weights, &features)?;
        error = error.saturating_add(byte_sample_probability_error_q15(
            &row.probabilities_q15,
            tokens[start + seq_len],
        ));
    }
    Ok(error)
}

fn byte_embed_total_error(
    tokens: &[u8],
    starts: &[usize],
    embeddings: &[i16],
    output_weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for &start in starts {
        let features = byte_embed_features_q15(embeddings, tokens, start, seq_len)?;
        let row = byte_embed_softmax_row_for(output_weights, &features)?;
        if byte_argmax_i32(&row.logits_q8) != tokens[start + seq_len] {
            mistakes += 1;
        }
    }
    Ok(mistakes)
}

fn byte_embed_total_probability_error_q15(
    tokens: &[u8],
    starts: &[usize],
    embeddings: &[i16],
    output_weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &start in starts {
        let features = byte_embed_features_q15(embeddings, tokens, start, seq_len)?;
        let row = byte_embed_softmax_row_for(output_weights, &features)?;
        error = error.saturating_add(byte_sample_probability_error_q15(
            &row.probabilities_q15,
            tokens[start + seq_len],
        ));
    }
    Ok(error)
}

fn mini_transformer_total_error(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for &start in starts {
        let cache = mini_transformer_forward_for(model, &tokens[start..start + seq_len])?;
        if byte_argmax_i32(&cache.logits_q8) != tokens[start + seq_len] {
            mistakes += 1;
        }
    }
    Ok(mistakes)
}

fn mini_transformer_total_probability_error_q15(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &start in starts {
        let cache = mini_transformer_forward_for(model, &tokens[start..start + seq_len])?;
        error = error.saturating_add(byte_sample_probability_error_q15(
            &cache.probabilities_q15,
            tokens[start + seq_len],
        ));
    }
    Ok(error)
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

fn hash_byte_logits(
    tokens: &[u8],
    starts: &[usize],
    weights: &[i8],
    seq_len: usize,
) -> Result<u64, TrainError> {
    let mut hasher = StableHasher::new();
    for &start in starts {
        let features = byte_features_q15(tokens, start, seq_len);
        let row = byte_softmax_row_for(weights, &features)?;
        hasher.update_i32_slice(&row.logits_q8);
    }
    Ok(hasher.finish())
}

fn hash_byte_embed_logits(
    tokens: &[u8],
    starts: &[usize],
    embeddings: &[i16],
    output_weights: &[i8],
    seq_len: usize,
) -> Result<u64, TrainError> {
    let mut hasher = StableHasher::new();
    for &start in starts {
        let features = byte_embed_features_q15(embeddings, tokens, start, seq_len)?;
        let row = byte_embed_softmax_row_for(output_weights, &features)?;
        hasher.update_i32_slice(&row.logits_q8);
    }
    Ok(hasher.finish())
}

fn hash_mini_transformer_logits(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<u64, TrainError> {
    let mut hasher = StableHasher::new();
    for &start in starts {
        let cache = mini_transformer_forward_for(model, &tokens[start..start + seq_len])?;
        hasher.update_i32_slice(&cache.logits_q8);
    }
    Ok(hasher.finish())
}

fn hash_byte_windows(tokens: &[u8], config: ByteSoftmaxTrainConfig, starts: &[usize]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.seq_len);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_bytes(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
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
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(MINI_TRANSFORMER_D_MODEL);
    hasher.update_usize(MINI_TRANSFORMER_HIDDEN_DIM);
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_bytes(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
}

fn hash_byte_embed_windows(
    tokens: &[u8],
    config: ByteEmbedSoftmaxTrainConfig,
    starts: &[usize],
) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.seq_len);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(BYTE_EMBED_DIM);
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_bytes(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoftmaxRow {
    logits_q8: [i32; VOCAB],
    probabilities_q15: [i16; VOCAB],
    softmax_sum_q15: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteSoftmaxRow {
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteEmbedSoftmaxRow {
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
}

fn softmax_row_for(weights: &[i8], input_id: usize) -> Result<SoftmaxRow, TrainError> {
    let logits = logits_for(weights, input_id)?;
    let mut logits_q8 = [0_i32; VOCAB];
    for (out, &logit) in logits_q8.iter_mut().zip(logits.iter()) {
        *out = i32::from(logit);
    }

    let mut probabilities_q15 = [0_i16; VOCAB];
    let softmax_sum_q15 = base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("base2_output_head_softmax"))?;

    Ok(SoftmaxRow {
        logits_q8,
        probabilities_q15,
        softmax_sum_q15,
    })
}

fn byte_softmax_row_for(
    weights: &[i8],
    features: &[i16; BYTE_D_MODEL],
) -> Result<ByteSoftmaxRow, TrainError> {
    let params = LinearI16I8Params {
        weights,
        bias: None,
        scales: &BYTE_LOGIT_SCALES,
        input_dim: BYTE_D_MODEL,
        output_dim: BYTE_VOCAB,
    };
    let mut logits = [0_i16; BYTE_VOCAB];
    linear_i16_i8_i16_per_channel_checked(features, params, &mut logits)
        .ok_or(TrainError::CoreRejected("byte_output_head_linear"))?;

    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (out, &logit) in logits_q8.iter_mut().zip(logits.iter()) {
        *out = i32::from(logit);
    }

    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("byte_output_head_softmax"))?;

    Ok(ByteSoftmaxRow {
        logits_q8,
        probabilities_q15,
    })
}

fn byte_embed_softmax_row_for(
    weights: &[i8],
    features: &[i16; BYTE_EMBED_D_MODEL],
) -> Result<ByteEmbedSoftmaxRow, TrainError> {
    let params = LinearI16I8Params {
        weights,
        bias: None,
        scales: &BYTE_EMBED_LOGIT_SCALES,
        input_dim: BYTE_EMBED_D_MODEL,
        output_dim: BYTE_VOCAB,
    };
    let mut logits = [0_i16; BYTE_VOCAB];
    linear_i16_i8_i16_per_channel_checked(features, params, &mut logits)
        .ok_or(TrainError::CoreRejected("byte_embed_output_head_linear"))?;

    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (out, &logit) in logits_q8.iter_mut().zip(logits.iter()) {
        *out = i32::from(logit);
    }

    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("byte_embed_output_head_softmax"))?;

    Ok(ByteEmbedSoftmaxRow {
        logits_q8,
        probabilities_q15,
    })
}

fn softmax_gradient_q15(probabilities_q15: &[i16; VOCAB], target_id: usize) -> [i32; VOCAB] {
    let mut gradient = [0_i32; VOCAB];
    for (class_id, out) in gradient.iter_mut().enumerate() {
        *out = i32::from(probabilities_q15[class_id]);
        if class_id == target_id {
            *out -= i32::from(i16::MAX);
        }
    }
    gradient
}

fn byte_softmax_gradient_q15(
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

fn byte_gradient_i32_to_i16(gradient: &[i32; BYTE_VOCAB]) -> [i16; BYTE_VOCAB] {
    let mut out = [0_i16; BYTE_VOCAB];
    for (dst, &src) in out.iter_mut().zip(gradient.iter()) {
        *dst = saturate_i16(i64::from(src));
    }
    out
}

fn argmax(logits: &[i16; VOCAB]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn argmax_i32(logits: &[i32; VOCAB]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
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

fn apply_softmax_output_head_update(
    weights: &mut [i8],
    input_id: usize,
    gradient_q15: &[i32; VOCAB],
    learning_rate: i32,
    learning_rate_shift: u8,
) -> SoftmaxUpdateStats {
    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;

    for (class_id, &gradient) in gradient_q15.iter().enumerate() {
        let row_start = class_id * D_MODEL;
        for (feature_index, &activation) in FEATURES_Q15[input_id].iter().enumerate() {
            if activation == 0 || gradient == 0 {
                continue;
            }

            let product = i64::from(gradient)
                .saturating_mul(i64::from(activation))
                .saturating_mul(i64::from(learning_rate));
            let scaled_gradient = round_shift_rhu_i64(product, learning_rate_shift);
            let delta = -scaled_gradient;
            if delta == 0 {
                zero_delta_count += 1;
            }

            let weight = &mut weights[row_start + feature_index];
            let previous = *weight;
            let unclamped = i64::from(previous).saturating_add(delta);
            let clamped = saturate_i8(unclamped);
            if i64::from(clamped) != unclamped {
                gradient_saturation_count += 1;
            }
            let applied_delta = i64::from(clamped) - i64::from(previous);
            weight_delta_l1 = weight_delta_l1.saturating_add(applied_delta.unsigned_abs());
            *weight = clamped;
        }
    }

    SoftmaxUpdateStats {
        gradient_saturation_count,
        zero_delta_count,
        weight_delta_l1,
    }
}

fn apply_byte_softmax_output_head_update(
    weights: &mut [i8],
    features: &[i16; BYTE_D_MODEL],
    gradient_q15: &[i32; BYTE_VOCAB],
    learning_rate: i32,
    learning_rate_shift: u8,
) -> SoftmaxUpdateStats {
    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;

    for (class_id, &gradient) in gradient_q15.iter().enumerate() {
        let row_start = class_id * BYTE_D_MODEL;
        for (feature_index, &activation) in features.iter().enumerate() {
            if activation == 0 || gradient == 0 {
                continue;
            }

            let product = i64::from(gradient)
                .saturating_mul(i64::from(activation))
                .saturating_mul(i64::from(learning_rate));
            let scaled_gradient = round_shift_rhu_i64(product, learning_rate_shift);
            let delta = -scaled_gradient;
            if delta == 0 {
                zero_delta_count += 1;
            }

            let weight = &mut weights[row_start + feature_index];
            let previous = *weight;
            let unclamped = i64::from(previous).saturating_add(delta);
            let clamped = saturate_i8(unclamped);
            if i64::from(clamped) != unclamped {
                gradient_saturation_count += 1;
            }
            let applied_delta = i64::from(clamped) - i64::from(previous);
            weight_delta_l1 = weight_delta_l1.saturating_add(applied_delta.unsigned_abs());
            *weight = clamped;
        }
    }

    SoftmaxUpdateStats {
        gradient_saturation_count,
        zero_delta_count,
        weight_delta_l1,
    }
}

fn apply_byte_embedding_update(
    embeddings: &mut [i16],
    context: &[u8],
    grad_embedding_features_q15: &[i16],
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
    seq_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != BYTE_VOCAB * BYTE_EMBED_DIM
        || context.is_empty()
        || grad_embedding_features_q15.len() != BYTE_EMBED_DIM
        || learning_rate <= 0
    {
        return Err(TrainError::InvalidConfig);
    }
    let total_shift = embedding_learning_rate_shift
        .checked_add(seq_shift)
        .ok_or(TrainError::InvalidConfig)?;
    if total_shift > MAX_RIGHT_SHIFT {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    };

    for &token in context {
        let row_start = usize::from(token) * BYTE_EMBED_DIM;
        for (dim, &gradient) in grad_embedding_features_q15.iter().enumerate() {
            if gradient == 0 {
                continue;
            }

            let product = i64::from(gradient).saturating_mul(i64::from(learning_rate));
            let scaled_update = round_shift_rhu_i64(product, total_shift);
            let delta = -scaled_update;
            if delta == 0 {
                stats.zero_delta_count += 1;
            }

            let embedding = &mut embeddings[row_start + dim];
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
    }

    Ok(stats)
}

fn apply_perceptron_update(
    weights: &mut [i8],
    input_id: usize,
    output_id: usize,
    learning_rate: i8,
    direction: i8,
) -> usize {
    let row_start = output_id * D_MODEL;
    let mut saturation_count = 0_usize;
    for (feature_index, &activation) in FEATURES_Q15[input_id].iter().enumerate() {
        let sign = activation.signum() as i8;
        if sign == 0 {
            continue;
        }

        let delta = i64::from(direction) * i64::from(learning_rate) * i64::from(sign);
        let weight = &mut weights[row_start + feature_index];
        let unclamped = i64::from(*weight) + delta;
        let clamped = saturate_i8(unclamped);
        if i64::from(clamped) != unclamped {
            saturation_count += 1;
        }
        *weight = clamped;
    }
    saturation_count
}

fn hash_logits(weights: &[i8]) -> Result<u64, TrainError> {
    let mut hasher = StableHasher::new();
    for input_id in 0..VOCAB {
        hasher.update_i16_slice(&logits_for(weights, input_id)?);
    }
    Ok(hasher.finish())
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

fn push_steps_field(out: &mut String, name: &str, steps: &[TrainingStepTrace]) {
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
        push_usize_field(out, "sample_index", step.sample_index);
        comma(out);
        push_usize_field(out, "input_id", step.input_id);
        comma(out);
        push_usize_field(out, "target_id", step.target_id);
        comma(out);
        push_usize_field(out, "predicted_id", step.predicted_id);
        comma(out);
        push_usize_field(out, "total_error_before", step.total_error_before);
        comma(out);
        push_usize_field(out, "total_error_after", step.total_error_after);
        comma(out);
        push_i32_field(out, "error_delta_i32", step.error_delta_i32);
        comma(out);
        push_hash_field(out, "weight_hash_before", step.weight_hash_before);
        comma(out);
        push_hash_field(out, "weight_hash_after", step.weight_hash_after);
        comma(out);
        push_usize_field(
            out,
            "gradient_saturation_count",
            step.gradient_saturation_count,
        );
        out.push('}');
    }
    out.push(']');
}

fn push_softmax_steps_field(out: &mut String, name: &str, steps: &[SoftmaxTrainingStepTrace]) {
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
        push_usize_field(out, "sample_index", step.sample_index);
        comma(out);
        push_usize_field(out, "input_id", step.input_id);
        comma(out);
        push_usize_field(out, "target_id", step.target_id);
        comma(out);
        push_usize_field(out, "predicted_id_before", step.predicted_id_before);
        comma(out);
        push_usize_field(out, "predicted_id_after", step.predicted_id_after);
        comma(out);
        push_i32_array_field(out, "logits_q8_before", &step.logits_q8_before);
        comma(out);
        push_i16_array_field(
            out,
            "probabilities_q15_before",
            &step.probabilities_q15_before,
        );
        comma(out);
        push_i32_array_field(out, "gradient_q15", &step.gradient_q15);
        comma(out);
        push_u64_field(out, "softmax_sum_q15_before", step.softmax_sum_q15_before);
        comma(out);
        push_usize_field(out, "total_error_before", step.total_error_before);
        comma(out);
        push_usize_field(out, "total_error_after", step.total_error_after);
        comma(out);
        push_i32_field(out, "error_delta_i32", step.error_delta_i32);
        comma(out);
        push_usize_field(
            out,
            "probability_error_before_q15",
            step.probability_error_before_q15,
        );
        comma(out);
        push_usize_field(
            out,
            "probability_error_after_q15",
            step.probability_error_after_q15,
        );
        comma(out);
        push_i32_field(
            out,
            "probability_error_delta_i32",
            step.probability_error_delta_i32,
        );
        comma(out);
        push_hash_field(out, "weight_hash_before", step.weight_hash_before);
        comma(out);
        push_hash_field(out, "weight_hash_after", step.weight_hash_after);
        comma(out);
        push_usize_field(
            out,
            "gradient_saturation_count",
            step.gradient_saturation_count,
        );
        comma(out);
        push_usize_field(out, "zero_delta_count", step.zero_delta_count);
        comma(out);
        push_u64_field(out, "weight_delta_l1", step.weight_delta_l1);
        out.push('}');
    }
    out.push(']');
}

fn push_byte_softmax_steps_field(
    out: &mut String,
    name: &str,
    steps: &[ByteSoftmaxTrainingStepTrace],
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
        push_hash_field(out, "weight_hash_before", step.weight_hash_before);
        comma(out);
        push_hash_field(out, "weight_hash_after", step.weight_hash_after);
        comma(out);
        push_usize_field(
            out,
            "gradient_saturation_count",
            step.gradient_saturation_count,
        );
        comma(out);
        push_usize_field(out, "zero_delta_count", step.zero_delta_count);
        comma(out);
        push_u64_field(out, "weight_delta_l1", step.weight_delta_l1);
        out.push('}');
    }
    out.push(']');
}

fn push_byte_embed_softmax_steps_field(
    out: &mut String,
    name: &str,
    steps: &[ByteEmbedSoftmaxTrainingStepTrace],
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
        push_hash_field(out, "embedding_hash_before", step.embedding_hash_before);
        comma(out);
        push_hash_field(out, "embedding_hash_after", step.embedding_hash_after);
        comma(out);
        push_hash_field(out, "weight_hash_before", step.weight_hash_before);
        comma(out);
        push_hash_field(out, "weight_hash_after", step.weight_hash_after);
        comma(out);
        push_usize_field(
            out,
            "gradient_saturation_count",
            step.gradient_saturation_count,
        );
        comma(out);
        push_usize_field(
            out,
            "embedding_saturation_count",
            step.embedding_saturation_count,
        );
        comma(out);
        push_usize_field(out, "zero_delta_count", step.zero_delta_count);
        comma(out);
        push_usize_field(
            out,
            "embedding_zero_delta_count",
            step.embedding_zero_delta_count,
        );
        comma(out);
        push_u64_field(out, "weight_delta_l1", step.weight_delta_l1);
        comma(out);
        push_u64_field(out, "embedding_delta_l1", step.embedding_delta_l1);
        out.push('}');
    }
    out.push(']');
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
            "attention_zero_delta_count",
            step.attention_zero_delta_count,
        );
        comma(out);
        push_u64_field(out, "output_head_delta_l1", step.output_head_delta_l1);
        comma(out);
        push_u64_field(out, "mlp_delta_l1", step.mlp_delta_l1);
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
        out.push('}');
    }
    out.push(']');
}

fn push_linear_weight_update_stats_field(
    out: &mut String,
    name: &str,
    stats: LinearWeightUpdateStats,
) {
    push_quoted(out, name);
    out.push_str(":{");
    push_usize_field(
        out,
        "gradient_saturation_count",
        stats.gradient_saturation_count,
    );
    comma(out);
    push_usize_field(out, "zero_delta_count", stats.zero_delta_count);
    comma(out);
    push_u64_field(out, "weight_delta_l1", stats.weight_delta_l1);
    out.push('}');
}

fn push_hash_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, &format!("0x{value:016x}"));
}

fn push_u64_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_usize_field(out: &mut String, name: &str, value: usize) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
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

fn push_i8_field(out: &mut String, name: &str, value: i8) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i16_field(out: &mut String, name: &str, value: i16) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i32_field(out: &mut String, name: &str, value: i32) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i16_array_field(out: &mut String, name: &str, values: &[i16]) {
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

fn push_i8_array_field(out: &mut String, name: &str, values: &[i8]) {
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

fn push_i32_array_field(out: &mut String, name: &str, values: &[i32]) {
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

fn push_fixed_scales_field(out: &mut String, name: &str, scales: &[FixedScale]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, scale) in scales.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push('{');
        push_i32_field(out, "multiplier", scale.multiplier);
        comma(out);
        push_usize_field(out, "right_shift", usize::from(scale.right_shift));
        out.push('}');
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
    fn training_smoke_reduces_mistakes_to_zero() {
        let trace = run_training_smoke(TrainConfig::default()).expect("train");

        assert!(trace.initial_mistakes > trace.final_mistakes);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert_eq!(trace.updates, 5);
        assert_eq!(trace.steps.len(), 5);
        assert_eq!(trace.gradient_saturation_count, 0);
        assert_eq!(trace.final_mistakes, 0);
        assert_eq!(trace.final_total_error, 0);
        assert_eq!(trace.final_accuracy_per_mille, 1000);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
    }

    #[test]
    fn training_trace_is_byte_stable() {
        let left = run_training_smoke(TrainConfig::default())
            .expect("left")
            .to_json_line();
        let right = run_training_smoke(TrainConfig::default())
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_smoke_trace.v1\""));
        assert!(left.contains("\"authority\":\"deterministic_training_replay\""));
        assert!(left.contains("\"steps\":["));
    }

    #[test]
    fn softmax_training_reduces_probability_error_and_converges() {
        let trace = run_softmax_training(SoftmaxTrainConfig::default()).expect("softmax train");

        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert_eq!(trace.final_total_error, 0);
        assert_eq!(trace.final_mistakes, 0);
        assert_eq!(trace.gradient_saturation_count, 0);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
        assert!(!trace.steps.is_empty());
        assert!(
            trace
                .steps
                .iter()
                .all(|step| step.gradient_saturation_count == 0)
        );
    }

    #[test]
    fn softmax_training_trace_is_byte_stable() {
        let left = run_softmax_training(SoftmaxTrainConfig::default())
            .expect("left")
            .to_json_line();
        let right = run_softmax_training(SoftmaxTrainConfig::default())
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_softmax_trace.v1\""));
        assert!(left.contains("\"gradient\":\"prob_q15_minus_target_q15\""));
        assert!(left.contains("\"probabilities_q15_before\""));
    }

    #[test]
    fn linear_backward_trace_updates_weights_and_transposes_gradient() {
        let trace = run_linear_backward_smoke(LinearBackwardConfig::default()).expect("linear bw");

        assert_eq!(trace.scaled_grad_output_i32, [8192, -6144, 512]);
        assert_eq!(trace.grad_input_q15, [3968, -5056, 832, -1472]);
        assert_eq!(trace.output_before_q15, [17408, -19968, 640]);
        assert_eq!(trace.output_after_q15, [-26112, 28416, -384]);
        assert_eq!(
            trace.weights_after_i8,
            [
                -5, 2, -1, 1, //
                5, 1, 1, 1, //
                1, 1, -3, 1,
            ]
        );
        assert_eq!(trace.update_stats.gradient_saturation_count, 0);
        assert_eq!(trace.update_stats.zero_delta_count, 3);
        assert_eq!(trace.update_stats.weight_delta_l1, 27);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
    }

    #[test]
    fn linear_backward_trace_is_byte_stable() {
        let left = run_linear_backward_smoke(LinearBackwardConfig::default())
            .expect("left")
            .to_json_line();
        let right = run_linear_backward_smoke(LinearBackwardConfig::default())
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_linear_backward_trace.v1\""));
        assert!(left.contains("\"intermediate\":\"i64_outer_product\""));
        assert!(left.contains("\"scaled_grad_output_i32\":[8192,-6144,512]"));
    }

    #[test]
    fn gated_mlp_backward_trace_updates_all_matrices() {
        let trace = run_gated_mlp_backward_smoke(LinearBackwardConfig {
            learning_rate: 1,
            learning_rate_shift: 20,
        })
        .expect("gated mlp bw");

        assert_ne!(trace.up_before_i8, trace.up_after_i8);
        assert_ne!(trace.gate_before_i8, trace.gate_after_i8);
        assert_ne!(trace.down_before_i8, trace.down_after_i8);
        assert_ne!(trace.output_before_q15, trace.output_after_q15);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
        assert_ne!(trace.output_hash_before, trace.output_hash_after);
        assert_eq!(trace.update_stats.gradient_saturation_count(), Some(0));
        assert!(trace.update_stats.weight_delta_l1().unwrap() > 0);
        assert!(trace.update_stats.up.weight_delta_l1 > 0);
        assert!(trace.update_stats.gate.weight_delta_l1 > 0);
        assert!(trace.update_stats.down.weight_delta_l1 > 0);
    }

    #[test]
    fn gated_mlp_backward_trace_is_byte_stable() {
        let config = LinearBackwardConfig {
            learning_rate: 1,
            learning_rate_shift: 20,
        };
        let left = run_gated_mlp_backward_smoke(config)
            .expect("left")
            .to_json_line();
        let right = run_gated_mlp_backward_smoke(config)
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_gated_mlp_backward_trace.v1\""));
        assert!(left.contains("\"trained_component\":\"gated_mlp_i8_matrices\""));
        assert!(left.contains("\"activation\":\"hard_silu_shift2_q15\""));
    }

    #[test]
    fn byte_softmax_training_learns_alternating_token_windows() {
        let tokens = b"abababababababab";
        let trace = run_byte_softmax_training(
            tokens,
            ByteSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 1,
                stride: 1,
                max_windows: Some(12),
                learning_rate: 1,
                learning_rate_shift: 25,
            },
        )
        .expect("byte train");

        assert_eq!(trace.token_count, tokens.len());
        assert_eq!(trace.windows, 12);
        assert_eq!(trace.updates, 24);
        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert_eq!(trace.final_total_error, 0);
        assert_eq!(trace.gradient_saturation_count, 0);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.target_probability_after_q15 > step.target_probability_before_q15)
        );
    }

    #[test]
    fn byte_softmax_training_trace_is_byte_stable() {
        let tokens = b"abababab";
        let config = ByteSoftmaxTrainConfig {
            epochs: 1,
            seq_len: 1,
            stride: 1,
            max_windows: Some(6),
            learning_rate: 1,
            learning_rate_shift: 25,
        };
        let left = run_byte_softmax_training(tokens, config)
            .expect("left")
            .to_json_line();
        let right = run_byte_softmax_training(tokens, config)
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_byte_softmax_trace.v1\""));
        assert!(left.contains("\"tokenizer\":\"byte_identity_u8_v1\""));
        assert!(left.contains("\"features\":\"bias_plus_last_byte_one_hot_q15\""));
    }

    #[test]
    fn byte_softmax_model_round_trips_and_generates() {
        let tokens = b"abababababababab";
        let run = run_byte_softmax_training_with_model(
            tokens,
            ByteSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 1,
                stride: 1,
                max_windows: Some(12),
                learning_rate: 1,
                learning_rate_shift: 25,
            },
        )
        .expect("train");
        let bytes = run.model.to_bytes();
        let decoded = ByteSoftmaxModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.weight_hash(), run.trace.final_weight_hash);

        let generation =
            generate_byte_softmax(&decoded, b"a", ByteGenerationConfig { max_new_tokens: 6 })
                .expect("generate");

        assert_eq!(generation.generated_bytes, b"bababa".to_vec());
        assert_eq!(generation.steps.len(), 6);
        assert!(
            generation
                .to_json_line()
                .contains("\"schema\":\"nsrl.byte_generation_trace.v1\"")
        );
        assert!(
            generation
                .to_json_line()
                .contains("\"model\":\"byte_softmax_bigram_output_head_v1\"")
        );
    }

    #[test]
    fn byte_embed_softmax_training_updates_embeddings_and_head() {
        let tokens = b"abababababababab";
        let trace = run_byte_embed_softmax_training(
            tokens,
            ByteEmbedSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 2,
                stride: 1,
                max_windows: Some(12),
                learning_rate: 1,
                head_learning_rate_shift: 17,
                embedding_learning_rate_shift: 0,
            },
        )
        .expect("byte embed train");

        assert_eq!(trace.token_count, tokens.len());
        assert_eq!(trace.windows, 12);
        assert_eq!(trace.updates, 24);
        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert_eq!(trace.gradient_saturation_count, 0);
        assert_eq!(trace.embedding_saturation_count, 0);
        assert!(trace.weight_delta_l1 > 0);
        assert!(trace.embedding_delta_l1 > 0);
        assert_ne!(trace.initial_embedding_hash, trace.final_embedding_hash);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.embedding_hash_before != step.embedding_hash_after)
        );
    }

    #[test]
    fn byte_embed_softmax_training_trace_is_byte_stable() {
        let tokens = b"abababab";
        let config = ByteEmbedSoftmaxTrainConfig {
            epochs: 1,
            seq_len: 2,
            stride: 1,
            max_windows: Some(5),
            learning_rate: 1,
            head_learning_rate_shift: 17,
            embedding_learning_rate_shift: 0,
        };
        let left = run_byte_embed_softmax_training(tokens, config)
            .expect("left")
            .to_json_line();
        let right = run_byte_embed_softmax_training(tokens, config)
            .expect("right")
            .to_json_line();

        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.training_byte_embed_softmax_trace.v1\""));
        assert!(left.contains("\"features\":\"bias_plus_mean_byte_embedding_q15\""));
        assert!(left.contains("\"embedding_dtype\":\"i16\""));
    }

    #[test]
    fn byte_embed_softmax_model_round_trips_and_generates() {
        let tokens = b"abababababababab";
        let run = run_byte_embed_softmax_training_with_model(
            tokens,
            ByteEmbedSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 2,
                stride: 1,
                max_windows: Some(12),
                learning_rate: 1,
                head_learning_rate_shift: 17,
                embedding_learning_rate_shift: 0,
            },
        )
        .expect("train");
        let bytes = run.model.to_bytes();
        let decoded = ByteEmbedSoftmaxModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.embedding_hash(), run.trace.final_embedding_hash);
        assert_eq!(decoded.output_weight_hash(), run.trace.final_weight_hash);

        let generation = generate_byte_embed_softmax(
            &decoded,
            b"ab",
            ByteGenerationConfig { max_new_tokens: 6 },
        )
        .expect("generate");

        assert_eq!(generation.generated_bytes.len(), 6);
        assert_eq!(generation.steps.len(), 6);
        assert_eq!(generation.context_seq_len, 2);
        assert_eq!(generation.model_hash, decoded.model_hash());
        assert!(
            generation
                .to_json_line()
                .contains("\"schema\":\"nsrl.byte_embed_generation_trace.v1\"")
        );
        assert!(
            generation
                .to_json_line()
                .contains("\"model\":\"byte_embed_softmax_context_head_v1\"")
        );
    }

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
                max_windows: Some(64),
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                attention_learning_rate_shift: 24,
                attention_qk_learning_rate_shift: 18,
            },
        )
        .expect("mini train");

        assert_eq!(trace.token_count, tokens.len());
        assert_eq!(trace.windows, 64);
        assert_eq!(trace.updates, 128);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert_ne!(trace.initial_model_hash, trace.final_model_hash);
        assert_ne!(trace.initial_output_head_hash, trace.final_output_head_hash);
        assert_ne!(trace.initial_mlp_hash, trace.final_mlp_hash);
        assert_ne!(trace.initial_attention_hash, trace.final_attention_hash);
        assert_ne!(trace.initial_attention_q_hash, trace.final_attention_q_hash);
        assert_ne!(trace.initial_attention_k_hash, trace.final_attention_k_hash);
        assert_ne!(trace.initial_attention_v_hash, trace.final_attention_v_hash);
        assert_ne!(trace.initial_attention_o_hash, trace.final_attention_o_hash);
        assert_eq!(trace.output_head_saturation_count, 0);
        assert_eq!(trace.attention_saturation_count, 0);
        assert!(trace.output_head_delta_l1 > 0);
        assert!(trace.mlp_delta_l1 > 0);
        assert!(trace.attention_delta_l1 > 0);
        assert!(trace.attention_q_delta_l1 > 0);
        assert!(trace.attention_k_delta_l1 > 0);
        assert!(trace.attention_v_delta_l1 > 0);
        assert!(trace.attention_o_delta_l1 > 0);
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
    fn mini_transformer_mlp_training_trace_is_byte_stable() {
        let tokens = b"abababababab";
        let config = MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            max_windows: Some(6),
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            attention_learning_rate_shift: 24,
            attention_qk_learning_rate_shift: 18,
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
        assert!(left.contains(
            "\"trained_component\":\"output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8\""
        ));
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
                max_windows: Some(32),
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                attention_learning_rate_shift: 24,
                attention_qk_learning_rate_shift: 18,
            },
        )
        .expect("mini train");
        let bytes = run.model.to_bytes();
        let decoded = MiniTransformerMlpModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.model_hash(), run.trace.final_model_hash);
        assert_eq!(decoded.embedding_hash(), run.model.embedding_hash());
        assert_eq!(decoded.attention_hash(), run.trace.final_attention_hash);
        assert_eq!(decoded.mlp_hash(), run.trace.final_mlp_hash);
        assert_eq!(decoded.output_head_hash(), run.trace.final_output_head_hash);

        let generation = generate_mini_transformer(
            &decoded,
            b"To be",
            ByteGenerationConfig { max_new_tokens: 8 },
        )
        .expect("generate");

        assert_eq!(generation.generated_bytes.len(), 8);
        assert_eq!(generation.steps.len(), 8);
        assert_eq!(generation.context_seq_len, decoded.context_seq_len);
        assert_eq!(generation.model_hash, decoded.model_hash());
        assert_eq!(generation.attention_hash, decoded.attention_hash());
        assert_eq!(generation.mlp_hash, decoded.mlp_hash());
        assert_eq!(generation.output_head_hash, decoded.output_head_hash());
        let line = generation.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.mini_transformer_generation_trace.v1\""));
        assert!(line.contains("\"model\":\"mini_transformer_byte_qkvo_mlp_v1\""));
    }
}
