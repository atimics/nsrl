#![deny(unsafe_code)]

use nsrl_core::{
    FixedScale, GatedMlpBackwardScales, GatedMlpBackwardWorkspace, GatedMlpI16Params,
    GatedMlpWeightUpdateParams, GatedMlpWeightUpdateStats, GatedMlpWeightUpdateWorkspace,
    GatedMlpWorkspace, LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace,
    LinearBackwardWeightUpdateI8Params, LinearBackwardWeightUpdateWorkspace, LinearI16I8Params,
    LinearWeightUpdateStats, MASKED_LOGIT, MAX_RIGHT_SHIFT, Q15_SHIFT, SelfAttentionI16Params,
    SelfAttentionWorkspace, attention_dot_q_k_i16_i32_checked, base2_softmax_i32_q15,
    gated_mlp_backward_input_i16_q15_checked, gated_mlp_backward_weight_update_i8_checked,
    gated_mlp_i16_q15_checked, linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_backward_prescale_grad_output_i16_i32_checked, linear_backward_weight_update_i8_checked,
    linear_i16_i8_i16_per_channel_checked, round_shift_rhu_i64, saturate_i8, saturate_i16,
    self_attention_i16_q15_checked, sqrt_power_of_four_shift,
};

pub const SCHEMA: &str = "nsrl.training_smoke_trace.v1";
pub const SOFTMAX_SCHEMA: &str = "nsrl.training_softmax_trace.v1";
pub const LINEAR_BACKWARD_SCHEMA: &str = "nsrl.training_linear_backward_trace.v1";
pub const GATED_MLP_BACKWARD_SCHEMA: &str = "nsrl.training_gated_mlp_backward_trace.v1";
pub const BYTE_SOFTMAX_SCHEMA: &str = "nsrl.training_byte_softmax_trace.v1";
pub const BYTE_GENERATION_SCHEMA: &str = "nsrl.byte_generation_trace.v1";
pub const BYTE_EMBED_SOFTMAX_SCHEMA: &str = "nsrl.training_byte_embed_softmax_trace.v1";
pub const BYTE_EMBED_GENERATION_SCHEMA: &str = "nsrl.byte_embed_generation_trace.v1";
pub const LEXEME_EMBEDDING_SCHEMA: &str = "nsrl.training_lexeme_embedding_trace.v1";
pub const LEXEME_SOFTMAX_SCHEMA: &str = "nsrl.training_lexeme_softmax_trace.v1";
pub const LEXEME_GENERATION_SCHEMA: &str = "nsrl.lexeme_generation_trace.v1";
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
pub const LEXEME_EMBEDDING_TASK: &str = "wiki_bard_lexeme_context_embedding_pretrain";
pub const LEXEME_SOFTMAX_TASK: &str = "wiki_bard_lexeme_next_token_output_head";
pub const MINI_TRANSFORMER_MLP_TASK: &str = "wiki_bard_mini_transformer_mlp_first";
pub const BYTE_TOKENIZER_ID: &str = "byte_identity_u8_v1";
pub const ASCII_LOWER_TOKENIZER_ID: &str = "byte_ascii_lower_text_u8_v1";
pub const LEXEME_TOKENIZER_ID: &str = "lexeme_ascii_lower_u16_v1";
pub const BYTE_SOFTMAX_MODEL_ID: &str = "byte_softmax_bigram_output_head_v1";
pub const BYTE_SOFTMAX_MODEL_MAGIC: &[u8; 8] = b"NSRLBM1\n";
pub const BYTE_EMBED_SOFTMAX_MODEL_ID: &str = "byte_embed_softmax_context_head_v1";
pub const BYTE_EMBED_SOFTMAX_MODEL_MAGIC: &[u8; 8] = b"NSRLEM1\n";
pub const LEXEME_EMBEDDING_MODEL_ID: &str = "lexeme_context_embedding_i16_v1";
pub const LEXEME_EMBEDDING_MODEL_MAGIC: &[u8; 8] = b"NSRLLX1\n";
pub const LEXEME_SOFTMAX_MODEL_ID: &str = "lexeme_softmax_embedding_head_v1";
pub const LEXEME_SOFTMAX_MODEL_MAGIC: &[u8; 8] = b"NSRLLM2\n";
pub const MINI_TRANSFORMER_MODEL_ID: &str = "mini_transformer_byte_qkvo_mlp_v1";
pub const MINI_TRANSFORMER_MODEL_MAGIC: &[u8; 8] = b"NSRLMT3\n";
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
pub const MINI_TRANSFORMER_D_MODEL: usize = 16;
pub const MINI_TRANSFORMER_HEADS: usize = 1;
pub const MINI_TRANSFORMER_HIDDEN_DIM: usize = 32;

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
const DEFAULT_LEXEME_EMBEDDING_EPOCHS: usize = 1;
const DEFAULT_LEXEME_EMBEDDING_CONTEXT_RADIUS: usize = 2;
const DEFAULT_LEXEME_EMBEDDING_STRIDE: usize = 1;
const DEFAULT_LEXEME_EMBEDDING_MAX_WINDOWS: usize = 4096;
const DEFAULT_LEXEME_EMBEDDING_VOCAB_SIZE: usize = 2048;
const DEFAULT_LEXEME_EMBEDDING_DIM: usize = 16;
const DEFAULT_LEXEME_EMBEDDING_LEARNING_RATE: i32 = 1;
const DEFAULT_LEXEME_EMBEDDING_LEARNING_RATE_SHIFT: u8 = 9;
const DEFAULT_LEXEME_FREQUENCY_WEIGHT_CAP: u32 = 0;
const DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15: i16 = 4096;
const LEXEME_POSITIVE_DOT_MARGIN_I64: i64 = 1_000_000;
const LEXEME_NEGATIVE_DOT_MARGIN_I64: i64 = 0;
const DEFAULT_LEXEME_SOFTMAX_EPOCHS: usize = 1;
const DEFAULT_LEXEME_SOFTMAX_SEQ_LEN: usize = 1;
const DEFAULT_LEXEME_SOFTMAX_STRIDE: usize = 1;
const DEFAULT_LEXEME_SOFTMAX_MAX_WINDOWS: usize = 4096;
const DEFAULT_LEXEME_SOFTMAX_LEARNING_RATE: i32 = 1;
const DEFAULT_LEXEME_SOFTMAX_LEARNING_RATE_SHIFT: u8 = 22;
const DEFAULT_LEXEME_SOFTMAX_MAX_WEIGHT_DELTA: i32 = 1;
const DEFAULT_LEXEME_SOFTMAX_LR_SHIFT_DECAY_WINDOWS: usize = 0;
const DEFAULT_LEXEME_SOFTMAX_LR_SHIFT_DECAY_STEP: u8 = 1;
const DEFAULT_LEXEME_SOFTMAX_MAX_LEARNING_RATE_SHIFT: u8 =
    DEFAULT_LEXEME_SOFTMAX_LEARNING_RATE_SHIFT;
const LEXEME_LOGIT_RIGHT_SHIFT: u8 = 8;
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
const DEFAULT_CORPUS_PRIOR_LOGIT_SHIFT: u8 = 8;
const MINI_TRANSFORMER_EMBEDDING_GRAD_FANIN_SHIFT: u8 = 1;
const MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT: usize = 8;
const PARALLEL_EVAL_MIN_ITEMS: usize = 512;
const PARALLEL_EVAL_MIN_ITEMS_PER_THREAD: usize = 128;
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
    "no_temperature_nucleus_or_beam_decode_yet",
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
    "no_temperature_nucleus_or_beam_decode_yet",
    "does_not_claim_language_model_quality",
];
const LEXEME_EMBEDDING_KNOWN_NON_CLAIMS: [&str; 5] = [
    "not_language_model_training_yet",
    "not_dynamic_vocabulary",
    "does_not_train_output_head_or_grammar",
    "negative_pairs_are_deterministic_not_sampled",
    "does_not_claim_semantic_quality",
];
const LEXEME_SOFTMAX_KNOWN_NON_CLAIMS: [&str; 5] = [
    "lexeme_output_head_only",
    "embeddings_are_frozen",
    "mean_pooled_context_not_attention",
    "not_full_transformer_backprop_yet",
    "does_not_claim_final_language_quality",
];
const LEXEME_GENERATION_KNOWN_NON_CLAIMS: [&str; 4] = [
    "lexeme_head_model_only",
    "mean_pooled_context_not_attention",
    "vocab_local_to_corpus",
    "not_final_spelling_or_grammar_layer",
];
const MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS: [&str; 5] = [
    "single_mini_transformer_block_only",
    "learned_absolute_position_embeddings_not_rope",
    "no_kv_cache_yet",
    "no_temperature_nucleus_or_beam_decode_yet",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS: [&str; 5] = [
    "single_mini_transformer_block_only",
    "embedding_table_updated_without_optimizer_state",
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
    pub window_offset: usize,
    pub max_windows: Option<usize>,
    pub tokenizer_id: ByteTokenizerId,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
}

impl Default for ByteSoftmaxTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_BYTE_SOFTMAX_EPOCHS,
            seq_len: DEFAULT_BYTE_SOFTMAX_SEQ_LEN,
            stride: DEFAULT_BYTE_SOFTMAX_STRIDE,
            window_offset: 0,
            max_windows: Some(DEFAULT_BYTE_SOFTMAX_MAX_WINDOWS),
            tokenizer_id: ByteTokenizerId::Identity,
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
    pub window_offset: usize,
    pub max_windows: Option<usize>,
    pub tokenizer_id: ByteTokenizerId,
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
            window_offset: 0,
            max_windows: Some(DEFAULT_BYTE_EMBED_SOFTMAX_MAX_WINDOWS),
            tokenizer_id: ByteTokenizerId::Identity,
            learning_rate: DEFAULT_BYTE_EMBED_SOFTMAX_LEARNING_RATE,
            head_learning_rate_shift: DEFAULT_BYTE_EMBED_SOFTMAX_HEAD_LEARNING_RATE_SHIFT,
            embedding_learning_rate_shift: DEFAULT_BYTE_EMBED_SOFTMAX_EMBEDDING_LEARNING_RATE_SHIFT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexemeEmbeddingTrainConfig {
    pub epochs: usize,
    pub context_radius: usize,
    pub stride: usize,
    pub window_offset: usize,
    pub max_windows: Option<usize>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
    pub concept_frequency_cap: u32,
    pub concept_frequency_min_weight_q15: i16,
    pub quality_weight_profile: LexemeQualityWeightProfile,
}

impl Default for LexemeEmbeddingTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_LEXEME_EMBEDDING_EPOCHS,
            context_radius: DEFAULT_LEXEME_EMBEDDING_CONTEXT_RADIUS,
            stride: DEFAULT_LEXEME_EMBEDDING_STRIDE,
            window_offset: 0,
            max_windows: Some(DEFAULT_LEXEME_EMBEDDING_MAX_WINDOWS),
            vocab_size: DEFAULT_LEXEME_EMBEDDING_VOCAB_SIZE,
            embedding_dim: DEFAULT_LEXEME_EMBEDDING_DIM,
            learning_rate: DEFAULT_LEXEME_EMBEDDING_LEARNING_RATE,
            learning_rate_shift: DEFAULT_LEXEME_EMBEDDING_LEARNING_RATE_SHIFT,
            concept_frequency_cap: DEFAULT_LEXEME_FREQUENCY_WEIGHT_CAP,
            concept_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            quality_weight_profile: LexemeQualityWeightProfile::Off,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexemeQualityWeightProfile {
    Off,
    CruftAware,
}

impl LexemeQualityWeightProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::CruftAware => "cruft-aware",
        }
    }
}

fn lexeme_quality_weight_profile_id(profile: LexemeQualityWeightProfile) -> usize {
    match profile {
        LexemeQualityWeightProfile::Off => 0,
        LexemeQualityWeightProfile::CruftAware => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexemeSoftmaxTrainConfig {
    pub epochs: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub window_offset: usize,
    pub max_windows: Option<usize>,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
    pub lr_shift_decay_windows: usize,
    pub lr_shift_decay_step: u8,
    pub max_learning_rate_shift: u8,
    pub max_weight_delta: i32,
    pub target_frequency_cap: u32,
    pub target_frequency_min_weight_q15: i16,
    pub quality_weight_profile: LexemeQualityWeightProfile,
}

impl Default for LexemeSoftmaxTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_LEXEME_SOFTMAX_EPOCHS,
            seq_len: DEFAULT_LEXEME_SOFTMAX_SEQ_LEN,
            stride: DEFAULT_LEXEME_SOFTMAX_STRIDE,
            window_offset: 0,
            max_windows: Some(DEFAULT_LEXEME_SOFTMAX_MAX_WINDOWS),
            learning_rate: DEFAULT_LEXEME_SOFTMAX_LEARNING_RATE,
            learning_rate_shift: DEFAULT_LEXEME_SOFTMAX_LEARNING_RATE_SHIFT,
            lr_shift_decay_windows: DEFAULT_LEXEME_SOFTMAX_LR_SHIFT_DECAY_WINDOWS,
            lr_shift_decay_step: DEFAULT_LEXEME_SOFTMAX_LR_SHIFT_DECAY_STEP,
            max_learning_rate_shift: DEFAULT_LEXEME_SOFTMAX_MAX_LEARNING_RATE_SHIFT,
            max_weight_delta: DEFAULT_LEXEME_SOFTMAX_MAX_WEIGHT_DELTA,
            target_frequency_cap: DEFAULT_LEXEME_FREQUENCY_WEIGHT_CAP,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            quality_weight_profile: LexemeQualityWeightProfile::Off,
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
    pub tokenizer_id: ByteTokenizerId,
    pub learning_rate: i32,
    pub output_learning_rate_shift: u8,
    pub mlp_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
    pub attention_learning_rate_shift: u8,
    pub attention_qk_learning_rate_shift: u8,
    pub attention_vo_error_feedback: bool,
    pub attention_vo_oracle: bool,
    pub reject_loss_regression: bool,
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
            tokenizer_id: ByteTokenizerId::Identity,
            learning_rate: DEFAULT_MINI_TRANSFORMER_LEARNING_RATE,
            output_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_HEAD_LEARNING_RATE_SHIFT,
            mlp_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_MLP_LEARNING_RATE_SHIFT,
            embedding_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_EMBEDDING_LEARNING_RATE_SHIFT,
            attention_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_ATTENTION_LEARNING_RATE_SHIFT,
            attention_qk_learning_rate_shift:
                DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
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
pub struct LexemeEmbeddingTrainingTrace {
    pub config: LexemeEmbeddingTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub positive_pair_count: usize,
    pub negative_pair_count: usize,
    pub initial_embedding_hash: u64,
    pub final_embedding_hash: u64,
    pub initial_positive_dot_i64: i64,
    pub final_positive_dot_i64: i64,
    pub initial_negative_dot_i64: i64,
    pub final_negative_dot_i64: i64,
    pub saturation_count: usize,
    pub zero_delta_count: usize,
    pub embedding_delta_l1: u64,
    pub steps: Vec<LexemeEmbeddingTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeEmbeddingTrainingRun {
    pub trace: LexemeEmbeddingTrainingTrace,
    pub model: LexemeEmbeddingModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeSoftmaxTrainingTrace {
    pub config: LexemeSoftmaxTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub vocab_size: usize,
    pub embedding_dim: usize,
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
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub steps: Vec<LexemeSoftmaxTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeSoftmaxTrainingRun {
    pub trace: LexemeSoftmaxTrainingTrace,
    pub model: LexemeSoftmaxModel,
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
pub struct LexemeEmbeddingModel {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub embeddings: Vec<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeSoftmaxModel {
    pub seq_len: usize,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub embeddings: Vec<i16>,
    pub output_weights: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpModel {
    pub context_seq_len: usize,
    pub embeddings: Vec<i16>,
    pub position_embeddings: Vec<i16>,
    pub q_weights: Vec<i8>,
    pub k_weights: Vec<i8>,
    pub v_weights: Vec<i8>,
    pub o_weights: Vec<i8>,
    pub up_weights: Vec<i8>,
    pub gate_weights: Vec<i8>,
    pub down_weights: Vec<i8>,
    pub output_weights: Vec<i8>,
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
pub struct LexemeGenerationConfig {
    pub max_new_tokens: usize,
    pub decode: DecodeConfig,
}

impl LexemeGenerationConfig {
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            decode: DecodeConfig::greedy(),
        }
    }

    pub fn deterministic_sample(max_new_tokens: usize, sample_seed: u64, top_k: usize) -> Self {
        Self {
            max_new_tokens,
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
    pub corpus_prior: bool,
    pub corpus_prior_logit_shift: u8,
    pub strict_adjacency: bool,
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
            corpus_prior: false,
            corpus_prior_logit_shift: DEFAULT_CORPUS_PRIOR_LOGIT_SHIFT,
            strict_adjacency: false,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeDecodePriors {
    pub token_count: usize,
    pub token_hash: u64,
    pub vocab_size: usize,
    unigram_counts: Vec<u32>,
    bigram_counts: Vec<u32>,
    row_totals: Vec<u32>,
    observed_bigrams: usize,
}

impl LexemeDecodePriors {
    pub fn from_tokens(tokens: &[u16], vocab_size: usize) -> Result<Self, TrainError> {
        if tokens.len() < 2 || vocab_size == 0 || vocab_size > usize::from(u16::MAX) + 1 {
            return Err(TrainError::InvalidConfig);
        }
        if tokens.iter().any(|&token| usize::from(token) >= vocab_size) {
            return Err(TrainError::InvalidConfig);
        }

        let mut unigram_counts = vec![0_u32; vocab_size];
        let mut bigram_counts = vec![
            0_u32;
            vocab_size
                .checked_mul(vocab_size)
                .ok_or(TrainError::InvalidConfig)?
        ];
        let mut row_totals = vec![0_u32; vocab_size];
        let mut observed_bigrams = 0_usize;

        for &token in tokens {
            let slot = usize::from(token);
            unigram_counts[slot] = unigram_counts[slot].saturating_add(1);
        }
        for pair in tokens.windows(2) {
            let previous = usize::from(pair[0]);
            let next = usize::from(pair[1]);
            let index = previous * vocab_size + next;
            if bigram_counts[index] == 0 {
                observed_bigrams = observed_bigrams.saturating_add(1);
            }
            bigram_counts[index] = bigram_counts[index].saturating_add(1);
            row_totals[previous] = row_totals[previous].saturating_add(1);
        }

        Ok(Self {
            token_count: tokens.len(),
            token_hash: hash_u16_slice(tokens),
            vocab_size,
            unigram_counts,
            bigram_counts,
            row_totals,
            observed_bigrams,
        })
    }

    pub fn observed_bigrams(&self) -> usize {
        self.observed_bigrams
    }

    pub fn transition_count(&self, previous: u16, next: u16) -> u32 {
        self.bigram_counts[usize::from(previous) * self.vocab_size + usize::from(next)]
    }

    pub fn allows_transition(&self, previous: u16, next: u16) -> bool {
        self.row_totals[usize::from(previous)] == 0 || self.transition_count(previous, next) > 0
    }

    pub fn transition_probability_q15(&self, previous: u16, next: u16) -> u16 {
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

    fn trace(&self) -> LexemeDecodePriorTrace {
        LexemeDecodePriorTrace {
            token_count: self.token_count,
            token_hash: self.token_hash,
            vocab_size: self.vocab_size,
            observed_bigrams: self.observed_bigrams,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexemeDecodePriorTrace {
    pub token_count: usize,
    pub token_hash: u64,
    pub vocab_size: usize,
    pub observed_bigrams: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeRejectStats {
    pub non_printable: usize,
    pub outside_ascii_lower: usize,
    pub byte_fallback: usize,
    pub repeat_run: usize,
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
struct LexemeDecodeSelection {
    token: u16,
    candidate_count: usize,
    rejected_candidates: DecodeRejectStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodeCandidateSet {
    candidates: Vec<usize>,
    rejected_candidates: DecodeRejectStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGenerationTrace {
    pub config: ByteGenerationConfig,
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub decode_priors: Option<ByteDecodePriorTrace>,
    pub steps: Vec<ByteGenerationStepTrace>,
}

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
pub struct ByteEmbedGenerationTrace {
    pub config: ByteGenerationConfig,
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub output_weight_hash: u64,
    pub context_seq_len: usize,
    pub decode_priors: Option<ByteDecodePriorTrace>,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerGenerationTrace {
    pub config: ByteGenerationConfig,
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
    pub context_seq_len: usize,
    pub decode_priors: Option<ByteDecodePriorTrace>,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeGenerationTrace {
    pub config: LexemeGenerationConfig,
    pub prompt_tokens: Vec<u16>,
    pub generated_tokens: Vec<u16>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub output_weight_hash: u64,
    pub context_seq_len: usize,
    pub decode_priors: Option<LexemeDecodePriorTrace>,
    pub steps: Vec<LexemeGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeGenerationStepTrace {
    pub step_index: usize,
    pub input_token: u16,
    pub predicted_token: u16,
    pub predicted_logit_q8: i32,
    pub predicted_probability_q15: i16,
    pub candidate_count: usize,
    pub rejected_candidates: DecodeRejectStats,
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
pub struct LexemeEmbeddingTrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub window_index: usize,
    pub center_index: usize,
    pub center_token: u16,
    pub context_token: u16,
    pub negative_token: u16,
    pub positive_frequency_weight_q15: i16,
    pub negative_frequency_weight_q15: i16,
    pub positive_quality_weight_q15: i16,
    pub negative_quality_weight_q15: i16,
    pub positive_update_weight_q15: i16,
    pub negative_update_weight_q15: i16,
    pub positive_dot_before_i64: i64,
    pub positive_dot_after_i64: i64,
    pub negative_dot_before_i64: i64,
    pub negative_dot_after_i64: i64,
    pub embedding_hash_before: u64,
    pub embedding_hash_after: u64,
    pub saturation_count: usize,
    pub zero_delta_count: usize,
    pub embedding_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeSoftmaxTrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub window_index: usize,
    pub previous_token: u16,
    pub target_token: u16,
    pub predicted_token_before: u16,
    pub predicted_token_after: u16,
    pub target_probability_before_q15: i16,
    pub target_probability_after_q15: i16,
    pub target_frequency_weight_q15: i16,
    pub target_quality_weight_q15: i16,
    pub target_update_weight_q15: i16,
    pub learning_rate_shift: u8,
    pub weight_hash_before: u64,
    pub weight_hash_after: u64,
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerMlpForwardCache {
    embedding_output: Vec<i16>,
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
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
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
        config.window_offset,
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
        config.window_offset,
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

pub fn run_lexeme_embedding_training(
    token_bytes: &[u8],
    config: LexemeEmbeddingTrainConfig,
) -> Result<LexemeEmbeddingTrainingTrace, TrainError> {
    Ok(run_lexeme_embedding_training_with_model(token_bytes, config)?.trace)
}

pub fn run_lexeme_embedding_training_with_model(
    token_bytes: &[u8],
    config: LexemeEmbeddingTrainConfig,
) -> Result<LexemeEmbeddingTrainingRun, TrainError> {
    run_lexeme_embedding_training_with_model_and_quality(token_bytes, config, None)
}

pub fn run_lexeme_embedding_training_with_model_and_quality(
    token_bytes: &[u8],
    config: LexemeEmbeddingTrainConfig,
    quality_weights_q15: Option<&[i16]>,
) -> Result<LexemeEmbeddingTrainingRun, TrainError> {
    if config.epochs == 0
        || config.context_radius == 0
        || config.stride == 0
        || config.vocab_size == 0
        || config.vocab_size > usize::from(u16::MAX) + 1
        || config.embedding_dim == 0
        || config.learning_rate <= 0
        || config.learning_rate_shift > MAX_RIGHT_SHIFT
        || !valid_q15_weight_floor(config.concept_frequency_min_weight_q15)
    {
        return Err(TrainError::InvalidConfig);
    }

    let tokens = decode_u16_tokens(token_bytes)?;
    if tokens.is_empty()
        || tokens
            .iter()
            .any(|&token| usize::from(token) >= config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }

    let starts = lexeme_center_starts(tokens.len(), config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u16_slice(&tokens);
    let window_hash = hash_lexeme_windows(&tokens, config, &starts);
    let concept_frequency_weights_q15 = lexeme_frequency_weights_q15(
        &tokens,
        config.vocab_size,
        config.concept_frequency_cap,
        config.concept_frequency_min_weight_q15,
    )?;
    let quality_weights_q15 = lexeme_training_quality_weights_q15(
        config.quality_weight_profile,
        quality_weights_q15,
        config.vocab_size,
    )?;
    let mut embeddings = initial_lexeme_embeddings(config.vocab_size, config.embedding_dim)?;
    let initial_embedding_hash = hash_i16_slice(&embeddings);
    let initial_positive_dot_i64 =
        lexeme_total_positive_dot_i64(&tokens, &starts, &embeddings, config);
    let initial_negative_dot_i64 =
        lexeme_total_negative_dot_i64(&tokens, &starts, &embeddings, config);

    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut positive_pair_count = 0_usize;
    let mut negative_pair_count = 0_usize;
    let mut saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut embedding_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (window_index, &center_index) in starts.iter().enumerate() {
            examined_windows = examined_windows.saturating_add(1);
            let center_token = tokens[center_index];
            let context_start = center_index - config.context_radius;
            let context_end = center_index + config.context_radius;

            for context_index in context_start..=context_end {
                if context_index == center_index {
                    continue;
                }
                let context_token = tokens[context_index];
                let negative_token =
                    lexeme_negative_token(center_token, context_token, updates, config.vocab_size);
                let positive_frequency_weight_q15 = lexeme_pair_frequency_weight_q15(
                    center_token,
                    context_token,
                    &concept_frequency_weights_q15,
                );
                let negative_frequency_weight_q15 = lexeme_pair_frequency_weight_q15(
                    center_token,
                    negative_token,
                    &concept_frequency_weights_q15,
                );
                let positive_quality_weight_q15 = lexeme_pair_frequency_weight_q15(
                    center_token,
                    context_token,
                    &quality_weights_q15,
                );
                let negative_quality_weight_q15 = lexeme_pair_frequency_weight_q15(
                    center_token,
                    negative_token,
                    &quality_weights_q15,
                );
                let positive_update_weight_q15 = lexeme_combine_q15_weights(
                    positive_frequency_weight_q15,
                    positive_quality_weight_q15,
                );
                let negative_update_weight_q15 = lexeme_combine_q15_weights(
                    negative_frequency_weight_q15,
                    negative_quality_weight_q15,
                );
                let positive_dot_before_i64 = lexeme_pair_dot_i64(
                    &embeddings,
                    config.embedding_dim,
                    center_token,
                    context_token,
                );
                let negative_dot_before_i64 = lexeme_pair_dot_i64(
                    &embeddings,
                    config.embedding_dim,
                    center_token,
                    negative_token,
                );
                let embedding_hash_before = hash_i16_slice(&embeddings);

                let positive_update = if positive_dot_before_i64 < LEXEME_POSITIVE_DOT_MARGIN_I64 {
                    apply_lexeme_embedding_pair_update(
                        &mut embeddings,
                        config.embedding_dim,
                        center_token,
                        context_token,
                        1,
                        config.learning_rate,
                        config.learning_rate_shift,
                        positive_update_weight_q15,
                    )?
                } else {
                    SoftmaxUpdateStats {
                        gradient_saturation_count: 0,
                        zero_delta_count: 0,
                        weight_delta_l1: 0,
                    }
                };
                let negative_update = if negative_dot_before_i64 > LEXEME_NEGATIVE_DOT_MARGIN_I64 {
                    apply_lexeme_embedding_pair_update(
                        &mut embeddings,
                        config.embedding_dim,
                        center_token,
                        negative_token,
                        -1,
                        config.learning_rate,
                        config.learning_rate_shift,
                        negative_update_weight_q15,
                    )?
                } else {
                    SoftmaxUpdateStats {
                        gradient_saturation_count: 0,
                        zero_delta_count: 0,
                        weight_delta_l1: 0,
                    }
                };

                updates = updates.saturating_add(1);
                positive_pair_count = positive_pair_count.saturating_add(1);
                negative_pair_count = negative_pair_count.saturating_add(1);
                saturation_count = saturation_count
                    .saturating_add(positive_update.gradient_saturation_count)
                    .saturating_add(negative_update.gradient_saturation_count);
                zero_delta_count = zero_delta_count
                    .saturating_add(positive_update.zero_delta_count)
                    .saturating_add(negative_update.zero_delta_count);
                embedding_delta_l1 = embedding_delta_l1
                    .saturating_add(positive_update.weight_delta_l1)
                    .saturating_add(negative_update.weight_delta_l1);

                if steps.len() < 16 {
                    let embedding_hash_after = hash_i16_slice(&embeddings);
                    let positive_dot_after_i64 = lexeme_pair_dot_i64(
                        &embeddings,
                        config.embedding_dim,
                        center_token,
                        context_token,
                    );
                    let negative_dot_after_i64 = lexeme_pair_dot_i64(
                        &embeddings,
                        config.embedding_dim,
                        center_token,
                        negative_token,
                    );
                    steps.push(LexemeEmbeddingTrainingStepTrace {
                        update_index: updates,
                        epoch,
                        window_index,
                        center_index,
                        center_token,
                        context_token,
                        negative_token,
                        positive_frequency_weight_q15,
                        negative_frequency_weight_q15,
                        positive_quality_weight_q15,
                        negative_quality_weight_q15,
                        positive_update_weight_q15,
                        negative_update_weight_q15,
                        positive_dot_before_i64,
                        positive_dot_after_i64,
                        negative_dot_before_i64,
                        negative_dot_after_i64,
                        embedding_hash_before,
                        embedding_hash_after,
                        saturation_count: positive_update
                            .gradient_saturation_count
                            .saturating_add(negative_update.gradient_saturation_count),
                        zero_delta_count: positive_update
                            .zero_delta_count
                            .saturating_add(negative_update.zero_delta_count),
                        embedding_delta_l1: positive_update
                            .weight_delta_l1
                            .saturating_add(negative_update.weight_delta_l1),
                    });
                }
            }
        }
    }

    let model = LexemeEmbeddingModel {
        vocab_size: config.vocab_size,
        embedding_dim: config.embedding_dim,
        embeddings,
    };
    let final_positive_dot_i64 =
        lexeme_total_positive_dot_i64(&tokens, &starts, &model.embeddings, config);
    let final_negative_dot_i64 =
        lexeme_total_negative_dot_i64(&tokens, &starts, &model.embeddings, config);
    let final_embedding_hash = model.embedding_hash();

    let trace = LexemeEmbeddingTrainingTrace {
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        positive_pair_count,
        negative_pair_count,
        initial_embedding_hash,
        final_embedding_hash,
        initial_positive_dot_i64,
        final_positive_dot_i64,
        initial_negative_dot_i64,
        final_negative_dot_i64,
        saturation_count,
        zero_delta_count,
        embedding_delta_l1,
        steps,
    };

    Ok(LexemeEmbeddingTrainingRun { trace, model })
}

pub fn run_lexeme_softmax_training(
    token_bytes: &[u8],
    embedding_model: LexemeEmbeddingModel,
    config: LexemeSoftmaxTrainConfig,
) -> Result<LexemeSoftmaxTrainingTrace, TrainError> {
    Ok(run_lexeme_softmax_training_with_model(token_bytes, embedding_model, config)?.trace)
}

pub fn run_lexeme_softmax_training_with_model(
    token_bytes: &[u8],
    embedding_model: LexemeEmbeddingModel,
    config: LexemeSoftmaxTrainConfig,
) -> Result<LexemeSoftmaxTrainingRun, TrainError> {
    run_lexeme_softmax_training_with_model_and_quality(token_bytes, embedding_model, config, None)
}

pub fn run_lexeme_softmax_training_with_model_and_quality(
    token_bytes: &[u8],
    embedding_model: LexemeEmbeddingModel,
    config: LexemeSoftmaxTrainConfig,
    quality_weights_q15: Option<&[i16]>,
) -> Result<LexemeSoftmaxTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || !config.seq_len.is_power_of_two()
        || config.stride == 0
        || config.learning_rate <= 0
        || config.learning_rate_shift > MAX_RIGHT_SHIFT
        || config.max_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.max_learning_rate_shift < config.learning_rate_shift
        || (config.lr_shift_decay_windows > 0 && config.lr_shift_decay_step == 0)
        || config.max_weight_delta <= 0
        || !valid_q15_weight_floor(config.target_frequency_min_weight_q15)
    {
        return Err(TrainError::InvalidConfig);
    }

    let tokens = decode_u16_tokens(token_bytes)?;
    if tokens.len() <= config.seq_len
        || tokens
            .iter()
            .any(|&token| usize::from(token) >= embedding_model.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }

    let starts = lexeme_softmax_starts(
        tokens.len(),
        config.seq_len,
        config.stride,
        config.window_offset,
        config.max_windows,
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let vocab_size = embedding_model.vocab_size;
    let embedding_dim = embedding_model.embedding_dim;
    let d_model = embedding_dim
        .checked_add(1)
        .ok_or(TrainError::InvalidConfig)?;
    let output_weight_count = vocab_size
        .checked_mul(d_model)
        .ok_or(TrainError::InvalidConfig)?;
    let mut output_weights = vec![0_i8; output_weight_count];
    let token_hash = hash_u16_slice(&tokens);
    let window_hash = hash_lexeme_softmax_windows(&tokens, config, &starts);
    let target_frequency_weights_q15 = lexeme_frequency_weights_q15(
        &tokens,
        vocab_size,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )?;
    let quality_weights_q15 = lexeme_training_quality_weights_q15(
        config.quality_weight_profile,
        quality_weights_q15,
        vocab_size,
    )?;
    let initial_embedding_hash = embedding_model.embedding_hash();
    let initial_weight_hash = hash_i8_slice(&output_weights);
    let initial_total_error = lexeme_total_error(
        &tokens,
        &starts,
        &embedding_model,
        &output_weights,
        config.seq_len,
    )?;
    let initial_probability_error_q15 = lexeme_total_probability_error_q15(
        &tokens,
        &starts,
        &embedding_model,
        &output_weights,
        config.seq_len,
    )?;
    let initial_mistakes = initial_total_error;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;
    let mut steps = Vec::new();

    for epoch in 0..config.epochs {
        for (window_index, &start) in starts.iter().enumerate() {
            examined_windows = examined_windows.saturating_add(1);
            let previous_token = tokens[start + config.seq_len - 1];
            let target_token = tokens[start + config.seq_len];
            let features = lexeme_context_features_q15(
                &embedding_model,
                &tokens[start..start + config.seq_len],
            )?;
            let row = lexeme_softmax_row_for(&output_weights, &features, vocab_size, d_model)?;
            let predicted_token_before = lexeme_argmax_i32(&row.logits_q8);
            let gradient_q15 = lexeme_softmax_gradient_q15(&row.probabilities_q15, target_token);
            let target_frequency_weight_q15 =
                target_frequency_weights_q15[usize::from(target_token)];
            let target_quality_weight_q15 = quality_weights_q15[usize::from(target_token)];
            let target_update_weight_q15 =
                lexeme_combine_q15_weights(target_frequency_weight_q15, target_quality_weight_q15);
            let weighted_gradient_q15 =
                lexeme_scale_gradient_q15(&gradient_q15, target_update_weight_q15);
            let weight_hash_before = hash_i8_slice(&output_weights);
            let learning_rate_shift =
                lexeme_softmax_learning_rate_shift_for_update(config, updates);
            let update = apply_lexeme_softmax_output_head_update(
                &mut output_weights,
                &features,
                &weighted_gradient_q15,
                vocab_size,
                d_model,
                config.learning_rate,
                learning_rate_shift,
                config.max_weight_delta,
            )?;
            updates = updates.saturating_add(1);
            gradient_saturation_count =
                gradient_saturation_count.saturating_add(update.gradient_saturation_count);
            zero_delta_count = zero_delta_count.saturating_add(update.zero_delta_count);
            weight_delta_l1 = weight_delta_l1.saturating_add(update.weight_delta_l1);

            if steps.len() < 16 {
                let after_row =
                    lexeme_softmax_row_for(&output_weights, &features, vocab_size, d_model)?;
                let predicted_token_after = lexeme_argmax_i32(&after_row.logits_q8);
                let weight_hash_after = hash_i8_slice(&output_weights);
                steps.push(LexemeSoftmaxTrainingStepTrace {
                    update_index: updates,
                    epoch,
                    window_index,
                    previous_token,
                    target_token,
                    predicted_token_before,
                    predicted_token_after,
                    target_probability_before_q15: row.probabilities_q15[usize::from(target_token)],
                    target_probability_after_q15: after_row.probabilities_q15
                        [usize::from(target_token)],
                    target_frequency_weight_q15,
                    target_quality_weight_q15,
                    target_update_weight_q15,
                    learning_rate_shift,
                    weight_hash_before,
                    weight_hash_after,
                    gradient_saturation_count: update.gradient_saturation_count,
                    zero_delta_count: update.zero_delta_count,
                    weight_delta_l1: update.weight_delta_l1,
                });
            }
        }
    }

    let final_total_error = lexeme_total_error(
        &tokens,
        &starts,
        &embedding_model,
        &output_weights,
        config.seq_len,
    )?;
    let final_probability_error_q15 = lexeme_total_probability_error_q15(
        &tokens,
        &starts,
        &embedding_model,
        &output_weights,
        config.seq_len,
    )?;
    let final_mistakes = final_total_error;
    let final_correct = starts.len().saturating_sub(final_mistakes);
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = hash_lexeme_logits(
        &tokens,
        &starts,
        &embedding_model,
        &output_weights,
        config.seq_len,
    )?;
    let model = LexemeSoftmaxModel::new(
        config.seq_len,
        embedding_model.vocab_size,
        embedding_model.embedding_dim,
        embedding_model.embeddings,
        output_weights,
    )?;
    let final_embedding_hash = model.embedding_hash();
    let final_weight_hash = model.output_weight_hash();

    let trace = LexemeSoftmaxTrainingTrace {
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        vocab_size,
        embedding_dim,
        initial_embedding_hash,
        final_embedding_hash,
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

    Ok(LexemeSoftmaxTrainingRun { trace, model })
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
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_training_from_model(tokens, config, model)
}

pub fn run_mini_transformer_mlp_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    mut model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.batch_windows == 0
        || config.learning_rate <= 0
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
        || (config.attention_vo_oracle && config.batch_windows <= 1)
    {
        return Err(TrainError::InvalidConfig);
    }
    mini_transformer_batch_learning_rate_shift(config.batch_windows)
        .ok_or(TrainError::InvalidConfig)?;
    if model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let starts = byte_window_starts(
        tokens.len(),
        config.seq_len,
        config.stride,
        config.window_offset,
        config.max_windows,
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let initial_model_hash = model.model_hash();
    let initial_embedding_hash = model.embedding_hash();
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
    let mut steps = Vec::new();
    let mut rollback_history = vec![model.clone()];
    let mut output_head_gradient =
        LinearWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, BYTE_VOCAB)
            .ok_or(TrainError::InvalidConfig)?;
    let mut mlp_weight_gradient =
        GatedMlpWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HIDDEN_DIM)
            .ok_or(TrainError::InvalidConfig)?;
    let mut attention_weight_gradient =
        MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidConfig)?;
    let mut embedding_gradient = MiniTransformerEmbeddingGradientI64::new(config.seq_len)
        .ok_or(TrainError::InvalidConfig)?;
    let use_output_head_accumulator = config.batch_windows > 1;
    let use_mlp_accumulator = config.batch_windows > 1;
    let use_attention_accumulator = config.batch_windows > 1;
    let use_embedding_accumulator = config.batch_windows > 1;

    for epoch in 0..config.epochs {
        let mut batch_start_index = 0_usize;
        while batch_start_index < starts.len() {
            let batch_end_index = batch_start_index
                .saturating_add(config.batch_windows)
                .min(starts.len());
            let batch_model_checkpoint = model.clone();
            let updates_before_batch = updates;
            let rollbacks_before_batch = rollback_count;

            for (relative_window_index, &window_start) in starts[batch_start_index..batch_end_index]
                .iter()
                .enumerate()
            {
                let window_index = batch_start_index + relative_window_index;
                examined_windows += 1;
                let target_token = tokens[window_start + config.seq_len];
                let cache_before = match mini_transformer_forward_for(
                    &model,
                    &tokens[window_start..window_start + config.seq_len],
                ) {
                    Ok(cache) => cache,
                    Err(_) => {
                        let mut recovered = None;
                        for checkpoint in rollback_history.iter().rev() {
                            if let Ok(cache) = mini_transformer_forward_for(
                                checkpoint,
                                &tokens[window_start..window_start + config.seq_len],
                            ) {
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
                                continue;
                            }
                        }
                    }
                };
                let predicted_token_before = byte_argmax_i32(&cache_before.logits_q8);
                let gradient_q15 =
                    byte_softmax_gradient_q15(&cache_before.probabilities_q15, target_token);
                let grad_output_q15 = byte_gradient_i32_to_i16(&gradient_q15);
                let output_head_hash_before = model.output_head_hash();
                let mlp_hash_before = model.mlp_hash();
                let attention_hash_before = model.attention_hash();
                let embedding_hash_before = model.embedding_hash();
                let model_checkpoint = model.clone();
                rollback_history.push(model_checkpoint.clone());
                if rollback_history.len() > MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT {
                    rollback_history.remove(0);
                }

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
                let output_update = if use_output_head_accumulator {
                    empty_linear_weight_update_stats()
                } else {
                    linear_backward_weight_update_i8_checked(
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
                    ))?
                };

                let mut grad_mlp_output = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                grad_mlp_output[last_start..last_end].copy_from_slice(&grad_last_features);
                let mut grad_mlp_input = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                let mut mlp_input_scaled_grad =
                    vec![0_i32; MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM)];
                let mut mlp_input_grad_gated =
                    vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
                let mut mlp_input_grad_up =
                    vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
                let mut mlp_input_grad_gate =
                    vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
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
                let grad_mlp_residual_input = grad_mlp_input;
                let mlp_rms_backward_saturation_count = 0_usize;

                let mut grad_attention_output =
                    vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                    &grad_mlp_output,
                    &grad_mlp_residual_input,
                    &mut grad_attention_output,
                )?;

                let mut mlp_scaled_grad =
                    vec![0_i32; MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM)];
                let mut grad_gated = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
                let mut grad_up = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
                let mut grad_gate = vec![0_i16; config.seq_len * MINI_TRANSFORMER_HIDDEN_DIM];
                let mlp_update = if use_mlp_accumulator {
                    empty_gated_mlp_weight_update_stats()
                } else {
                    gated_mlp_backward_weight_update_i8_checked(
                        &cache_before.mlp_norm,
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
                    .ok_or(TrainError::CoreRejected("mini_transformer_mlp_update"))?
                };

                let attention_update = mini_transformer_attention_update_i8_checked(
                    &cache_before,
                    &grad_attention_output,
                    &mut model,
                    config,
                    if use_attention_accumulator {
                        Some(&mut attention_weight_gradient)
                    } else {
                        None
                    },
                )?;
                let grad_attention_norm_input = attention_update.grad_embedding_output;
                let attention_rms_backward_saturation_count = 0_usize;

                let mut grad_embedding_output =
                    vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                let embedding_gradient_saturation_count = add_i16_residual_rows_checked(
                    &grad_attention_output,
                    &grad_attention_norm_input,
                    &mut grad_embedding_output,
                )?;
                let embedding_update = if use_embedding_accumulator {
                    empty_softmax_update_stats()
                } else {
                    apply_mini_transformer_embedding_update(
                        &mut model.embeddings,
                        &mut model.position_embeddings,
                        &tokens[window_start..window_start + config.seq_len],
                        &grad_embedding_output,
                        config.learning_rate,
                        config.embedding_learning_rate_shift,
                    )?
                };

                let cache_after = match mini_transformer_forward_for(
                    &model,
                    &tokens[window_start..window_start + config.seq_len],
                ) {
                    Ok(cache) => cache,
                    Err(_) => {
                        model = model_checkpoint;
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count = rejected_window_count.saturating_add(1);
                        continue;
                    }
                };

                if mini_transformer_validate_guard_windows(
                    &model,
                    tokens,
                    &starts,
                    config.seq_len,
                    epoch,
                    window_index,
                    config.epochs,
                )
                .is_err()
                {
                    model = model_checkpoint;
                    rollback_count = rollback_count.saturating_add(1);
                    rejected_window_count = rejected_window_count.saturating_add(1);
                    continue;
                }
                if use_output_head_accumulator {
                    accumulate_linear_weight_gradient_i64_prescaled(
                        &cache_before.block_output[last_start..last_end],
                        &output_scaled_grad,
                        &mut output_head_gradient,
                    )?;
                }
                if use_mlp_accumulator {
                    accumulate_gated_mlp_weight_gradient_i64(
                        &cache_before.mlp_norm,
                        &grad_mlp_output,
                        &cache_before.mlp_gated,
                        &mlp_input_grad_up,
                        &mlp_input_grad_gate,
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
                        &mut mlp_weight_gradient,
                        &mut mlp_scaled_grad,
                    )?;
                }
                if use_embedding_accumulator {
                    accumulate_mini_transformer_embedding_gradient_i64(
                        &tokens[window_start..window_start + config.seq_len],
                        &grad_embedding_output,
                        &mut embedding_gradient,
                    )?;
                }
                let predicted_token_after = byte_argmax_i32(&cache_after.logits_q8);
                let output_head_hash_after = model.output_head_hash();
                let mlp_hash_after = model.mlp_hash();
                let attention_hash_after = model.attention_hash();
                let embedding_hash_after = model.embedding_hash();

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
                    mlp_zero_delta_count: mlp_update.zero_delta_count().unwrap_or(usize::MAX),
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
                    let mlp_weight_gradient_checkpoint = mlp_weight_gradient.clone();
                    let attention_weight_gradient_checkpoint = attention_weight_gradient.clone();
                    let embedding_gradient_checkpoint = embedding_gradient.clone();
                    let batch_windows = &starts[batch_start_index..batch_end_index];

                    if use_output_head_accumulator {
                        let output_batch_update = apply_linear_weight_gradient_i64_to_i8(
                            &mut output_head_gradient,
                            &mut candidate_model.output_weights,
                            config.learning_rate,
                            config.output_learning_rate_shift,
                            false,
                        )?;
                        batch_output_head_saturation_count =
                            output_batch_update.gradient_saturation_count;
                        batch_output_head_zero_delta_count = output_batch_update.zero_delta_count;
                        batch_output_head_delta_l1 = output_batch_update.weight_delta_l1;
                        batch_output_head_accumulator_batch_count = 1;
                        batch_output_head_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_mlp_accumulator {
                        let mlp_batch_update = apply_gated_mlp_weight_gradient_i64_to_i8(
                            &mut mlp_weight_gradient,
                            &mut candidate_model.up_weights,
                            &mut candidate_model.gate_weights,
                            &mut candidate_model.down_weights,
                            config.learning_rate,
                            config.mlp_learning_rate_shift,
                        )?;
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
                            apply_mini_transformer_attention_weight_gradient_i64_to_i8(
                                &mut attention_weight_gradient,
                                &mut candidate_model,
                                config,
                            )?;
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
                        batch_attention_accumulator_batch_count = 1;
                        batch_attention_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_embedding_accumulator {
                        let embedding_batch_update =
                            apply_mini_transformer_embedding_gradient_i64_to_i16(
                                &mut embedding_gradient,
                                &mut candidate_model.embeddings,
                                &mut candidate_model.position_embeddings,
                                config.learning_rate,
                                config.embedding_learning_rate_shift,
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
                    )
                    .and_then(|_| {
                        mini_transformer_validate_guard_windows(
                            &candidate_model,
                            tokens,
                            &starts,
                            config.seq_len,
                            epoch,
                            batch_end_index.saturating_sub(1).min(starts.len() - 1),
                            config.epochs,
                        )
                    })
                    .is_ok();
                    let mut batch_loss_regressed = false;
                    if batch_valid && config.reject_loss_regression {
                        let before_loss = mini_transformer_total_probability_error_q15(
                            tokens,
                            &starts,
                            &batch_model_checkpoint,
                            config.seq_len,
                        )?;
                        let after_loss = mini_transformer_total_probability_error_q15(
                            tokens,
                            &starts,
                            &candidate_model,
                            config.seq_len,
                        )?;
                        batch_loss_regressed = after_loss > before_loss;
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
                        accepted_batch_count = accepted_batch_count.saturating_add(1);
                    } else {
                        model = batch_model_checkpoint;
                        updates = updates_before_batch;
                        steps.truncate(updates_before_batch);
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count =
                            rejected_window_count.saturating_add(accepted_windows_in_batch);
                        rejected_batch_count = rejected_batch_count.saturating_add(1);
                        if batch_loss_regressed {
                            loss_regression_rejected_batch_count =
                                loss_regression_rejected_batch_count.saturating_add(1);
                        }
                        if use_output_head_accumulator {
                            output_head_gradient = output_head_gradient_checkpoint;
                            output_head_gradient.clear();
                        }
                        if use_mlp_accumulator {
                            mlp_weight_gradient = mlp_weight_gradient_checkpoint;
                            mlp_weight_gradient.clear();
                        }
                        if use_attention_accumulator {
                            attention_weight_gradient = attention_weight_gradient_checkpoint;
                            attention_weight_gradient.clear();
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
                        mlp_weight_gradient.clear();
                    }
                    if use_attention_accumulator {
                        attention_weight_gradient.clear();
                    }
                    if use_embedding_accumulator {
                        embedding_gradient.clear();
                    }
                    accepted_batch_count = accepted_batch_count.saturating_add(1);
                }
            } else {
                if use_output_head_accumulator {
                    output_head_gradient.clear();
                }
                if use_mlp_accumulator {
                    mlp_weight_gradient.clear();
                }
                if use_attention_accumulator {
                    attention_weight_gradient.clear();
                }
                if use_embedding_accumulator {
                    embedding_gradient.clear();
                }
                rejected_batch_count = rejected_batch_count.saturating_add(1);
            }
            batch_start_index = batch_end_index;
        }
    }

    let final_eval = mini_transformer_eval_summary(tokens, &starts, &model, config.seq_len)?;
    let final_total_error = final_eval.mistakes;
    let final_probability_error_q15 = final_eval.probability_error_q15;
    let final_mistakes = final_eval.mistakes;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = final_eval.logits_hash;

    let trace = MiniTransformerMlpTrainingTrace {
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
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
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
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
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

impl LexemeEmbeddingTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", LEXEME_EMBEDDING_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", LEXEME_EMBEDDING_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", LEXEME_TOKENIZER_ID);
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
        push_string_field(&mut out, "id", LEXEME_EMBEDDING_MODEL_ID);
        comma(&mut out);
        push_usize_field(&mut out, "vocab_size", self.config.vocab_size);
        comma(&mut out);
        push_usize_field(&mut out, "embedding_dim", self.config.embedding_dim);
        comma(&mut out);
        push_string_field(&mut out, "trained_component", "lexeme_embedding_i16");
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "deterministic_integer_skipgram_hinge");
        comma(&mut out);
        push_string_field(&mut out, "embedding_scale", "q15_i16");
        comma(&mut out);
        push_string_field(&mut out, "positive_update", "pull_context_pair");
        comma(&mut out);
        push_string_field(
            &mut out,
            "negative_update",
            "push_deterministic_negative_pair",
        );
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "learning_rate_shift",
            usize::from(self.config.learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "concept_frequency_cap",
            self.config.concept_frequency_cap as usize,
        );
        comma(&mut out);
        push_i16_field(
            &mut out,
            "concept_frequency_min_weight_q15",
            self.config.concept_frequency_min_weight_q15,
        );
        comma(&mut out);
        push_string_field(
            &mut out,
            "quality_weight_profile",
            self.config.quality_weight_profile.as_str(),
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "positive_dot_margin_i64",
            LEXEME_POSITIVE_DOT_MARGIN_I64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "negative_dot_margin_i64",
            LEXEME_NEGATIVE_DOT_MARGIN_I64,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "context_radius", self.config.context_radius);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        comma(&mut out);
        push_usize_field(&mut out, "positive_pair_count", self.positive_pair_count);
        comma(&mut out);
        push_usize_field(&mut out, "negative_pair_count", self.negative_pair_count);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_i64_field(
            &mut out,
            "initial_positive_dot_i64",
            self.initial_positive_dot_i64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "final_positive_dot_i64",
            self.final_positive_dot_i64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "positive_dot_delta_i64",
            self.final_positive_dot_i64 - self.initial_positive_dot_i64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "initial_negative_dot_i64",
            self.initial_negative_dot_i64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "final_negative_dot_i64",
            self.final_negative_dot_i64,
        );
        comma(&mut out);
        push_i64_field(
            &mut out,
            "negative_dot_delta_i64",
            self.final_negative_dot_i64 - self.initial_negative_dot_i64,
        );
        comma(&mut out);
        push_usize_field(&mut out, "saturation_count", self.saturation_count);
        comma(&mut out);
        push_usize_field(&mut out, "zero_delta_count", self.zero_delta_count);
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
        push_lexeme_embedding_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &LEXEME_EMBEDDING_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl LexemeSoftmaxTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", LEXEME_SOFTMAX_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", LEXEME_SOFTMAX_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", LEXEME_TOKENIZER_ID);
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
        push_string_field(&mut out, "id", LEXEME_SOFTMAX_MODEL_ID);
        comma(&mut out);
        push_usize_field(&mut out, "vocab_size", self.vocab_size);
        comma(&mut out);
        push_usize_field(&mut out, "embedding_dim", self.embedding_dim);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", self.embedding_dim + 1);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.config.seq_len);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trained_component",
            "frozen_lexeme_embedding_i16_plus_output_head_i8",
        );
        comma(&mut out);
        push_string_field(
            &mut out,
            "features",
            "bias_plus_mean_context_lexeme_embedding_q15",
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "logit_scale", "q15_times_i8_shift8");
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
        push_usize_field(
            &mut out,
            "lr_shift_decay_windows",
            self.config.lr_shift_decay_windows,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "lr_shift_decay_step",
            usize::from(self.config.lr_shift_decay_step),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "max_learning_rate_shift",
            usize::from(self.config.max_learning_rate_shift),
        );
        comma(&mut out);
        push_i32_field(&mut out, "max_weight_delta", self.config.max_weight_delta);
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
        push_string_field(
            &mut out,
            "quality_weight_profile",
            self.config.quality_weight_profile.as_str(),
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
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
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
        push_i64_field(
            &mut out,
            "probability_error_delta_i64",
            self.final_probability_error_q15 as i64 - self.initial_probability_error_q15 as i64,
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
        push_lexeme_softmax_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &LEXEME_SOFTMAX_KNOWN_NON_CLAIMS,
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
        push_string_field(&mut out, "position", "learned_absolute_i16");
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
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
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

impl LexemeEmbeddingModel {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        embeddings: Vec<i16>,
    ) -> Result<Self, TrainError> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(TrainError::InvalidModel("bad lexeme embedding shape"));
        }
        if embeddings.len()
            != vocab_size
                .checked_mul(embedding_dim)
                .ok_or(TrainError::InvalidModel("lexeme embedding count overflow"))?
        {
            return Err(TrainError::InvalidModel("wrong lexeme embedding count"));
        }
        Ok(Self {
            vocab_size,
            embedding_dim,
            embeddings,
        })
    }

    pub fn embedding_hash(&self) -> u64 {
        hash_i16_slice(&self.embeddings)
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.vocab_size);
        hasher.update_usize(self.embedding_dim);
        hasher.update_i16_slice(&self.embeddings);
        hasher.finish()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36 + self.embeddings.len() * 2);
        out.extend_from_slice(LEXEME_EMBEDDING_MODEL_MAGIC);
        out.extend_from_slice(&(self.vocab_size as u32).to_le_bytes());
        out.extend_from_slice(&(self.embedding_dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.embeddings.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.embedding_hash().to_le_bytes());
        for &embedding in self.embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = LEXEME_EMBEDDING_MODEL_MAGIC.len() + 4 + 4 + 8 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        if &bytes[..LEXEME_EMBEDDING_MODEL_MAGIC.len()] != LEXEME_EMBEDDING_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = LEXEME_EMBEDDING_MODEL_MAGIC.len();
        let vocab_size = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_dim = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_embedding_hash = read_u64_le(bytes, &mut offset)?;
        let embedding_bytes = embedding_count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("embedding length overflow"))?;
        let embedding_end = offset
            .checked_add(embedding_bytes)
            .ok_or(TrainError::InvalidModel("embedding offset overflow"))?;
        if bytes.len() != embedding_end {
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
        let model = Self::new(vocab_size, embedding_dim, embeddings)?;
        if model.embedding_hash() != expected_embedding_hash {
            return Err(TrainError::InvalidModel("embedding hash mismatch"));
        }
        Ok(model)
    }
}

impl LexemeSoftmaxModel {
    pub fn new(
        seq_len: usize,
        vocab_size: usize,
        embedding_dim: usize,
        embeddings: Vec<i16>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        if seq_len == 0 || !seq_len.is_power_of_two() || vocab_size == 0 || embedding_dim == 0 {
            return Err(TrainError::InvalidModel("bad lexeme softmax shape"));
        }
        let embedding_count = vocab_size
            .checked_mul(embedding_dim)
            .ok_or(TrainError::InvalidModel("lexeme embedding count overflow"))?;
        let d_model = embedding_dim
            .checked_add(1)
            .ok_or(TrainError::InvalidModel("lexeme d_model overflow"))?;
        let weight_count = vocab_size
            .checked_mul(d_model)
            .ok_or(TrainError::InvalidModel("lexeme weight count overflow"))?;
        if embeddings.len() != embedding_count {
            return Err(TrainError::InvalidModel("wrong lexeme embedding count"));
        }
        if output_weights.len() != weight_count {
            return Err(TrainError::InvalidModel("wrong lexeme output weight count"));
        }
        Ok(Self {
            seq_len,
            vocab_size,
            embedding_dim,
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
        hasher.update_usize(self.vocab_size);
        hasher.update_usize(self.embedding_dim);
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i8_slice(&self.output_weights);
        hasher.finish()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(52 + self.embeddings.len() * 2 + self.output_weights.len());
        out.extend_from_slice(LEXEME_SOFTMAX_MODEL_MAGIC);
        out.extend_from_slice(&(self.seq_len as u32).to_le_bytes());
        out.extend_from_slice(&(self.vocab_size as u32).to_le_bytes());
        out.extend_from_slice(&(self.embedding_dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.embeddings.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.output_weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.embedding_hash().to_le_bytes());
        out.extend_from_slice(&self.output_weight_hash().to_le_bytes());
        out.extend_from_slice(&self.model_hash().to_le_bytes());
        for &embedding in self.embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        out.extend(self.output_weights.iter().map(|&weight| weight as u8));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = LEXEME_SOFTMAX_MODEL_MAGIC.len() + 4 + 4 + 4 + 8 + 8 + 8 + 8 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        if &bytes[..LEXEME_SOFTMAX_MODEL_MAGIC.len()] != LEXEME_SOFTMAX_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = LEXEME_SOFTMAX_MODEL_MAGIC.len();
        let seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let vocab_size = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_dim = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let weight_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_embedding_hash = read_u64_le(bytes, &mut offset)?;
        let expected_weight_hash = read_u64_le(bytes, &mut offset)?;
        let expected_model_hash = read_u64_le(bytes, &mut offset)?;

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
        let model = Self::new(
            seq_len,
            vocab_size,
            embedding_dim,
            embeddings,
            output_weights,
        )?;
        if model.embedding_hash() != expected_embedding_hash {
            return Err(TrainError::InvalidModel("embedding hash mismatch"));
        }
        if model.output_weight_hash() != expected_weight_hash {
            return Err(TrainError::InvalidModel("weight hash mismatch"));
        }
        if model.model_hash() != expected_model_hash {
            return Err(TrainError::InvalidModel("model hash mismatch"));
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
            position_embeddings: initial_mini_transformer_position_embeddings(context_seq_len),
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
            position_embeddings,
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
        let position_embedding_bytes = self.position_embeddings.len() * 2;
        let weight_bytes = self.q_weights.len()
            + self.k_weights.len()
            + self.v_weights.len()
            + self.o_weights.len()
            + self.up_weights.len()
            + self.gate_weights.len()
            + self.down_weights.len()
            + self.output_weights.len();
        let mut out =
            Vec::with_capacity(136 + embedding_bytes + position_embedding_bytes + weight_bytes);
        out.extend_from_slice(MINI_TRANSFORMER_MODEL_MAGIC);
        out.extend_from_slice(&(BYTE_VOCAB as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_D_MODEL as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_HEADS as u32).to_le_bytes());
        out.extend_from_slice(&(MINI_TRANSFORMER_HIDDEN_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(self.context_seq_len as u32).to_le_bytes());
        out.extend_from_slice(&(self.embeddings.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.position_embeddings.len() as u64).to_le_bytes());
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
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 8;
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

        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }

        let expected_attention_count = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL;
        let expected_position_embedding_count = context_seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel(
                "position embedding count overflow",
            ))?;
        if embedding_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
            || position_embedding_count != expected_position_embedding_count
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
        let expected_len = offset
            .checked_add(embedding_bytes)
            .and_then(|value| value.checked_add(position_embedding_bytes))
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

        let model = Self::new(
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
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
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
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
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

impl LexemeGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", LEXEME_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", LEXEME_SOFTMAX_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", LEXEME_TOKENIZER_ID);
        comma(&mut out);
        push_lexeme_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_lexeme_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
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
        push_usize_field(&mut out, "tokens_len", self.prompt_tokens.len());
        comma(&mut out);
        push_u16_array_field(&mut out, "tokens", &self.prompt_tokens);
        out.push('}');
        comma(&mut out);
        out.push_str("\"generation\":{");
        push_usize_field(&mut out, "new_tokens", self.generated_tokens.len());
        comma(&mut out);
        push_u16_array_field(&mut out, "tokens", &self.generated_tokens);
        out.push('}');
        comma(&mut out);
        push_lexeme_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &LEXEME_GENERATION_KNOWN_NON_CLAIMS,
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
    generate_byte_softmax_with_priors(model, prompt, config, None)
}

pub fn generate_byte_softmax_with_priors(
    model: &ByteSoftmaxModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<ByteGenerationTrace, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let features = byte_single_token_features_q15(input_token);
        let row = byte_softmax_row_for(&model.weights, &features)?;
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

    Ok(ByteGenerationTrace {
        config,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.weight_hash(),
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        steps,
    })
}

pub fn generate_byte_embed_softmax(
    model: &ByteEmbedSoftmaxModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<ByteEmbedGenerationTrace, TrainError> {
    generate_byte_embed_softmax_with_priors(model, prompt, config, None)
}

pub fn generate_byte_embed_softmax_with_priors(
    model: &ByteEmbedSoftmaxModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<ByteEmbedGenerationTrace, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    byte_embed_seq_shift(model.seq_len)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let features = byte_embed_context_features_q15(&model.embeddings, &context, model.seq_len)?;
        let row = byte_embed_softmax_row_for(&model.output_weights, &features)?;
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

    Ok(ByteEmbedGenerationTrace {
        config,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        output_weight_hash: model.output_weight_hash(),
        context_seq_len: model.seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        steps,
    })
}

pub fn generate_lexeme_softmax(
    model: &LexemeSoftmaxModel,
    prompt: &[u16],
    config: LexemeGenerationConfig,
) -> Result<LexemeGenerationTrace, TrainError> {
    generate_lexeme_softmax_with_priors(model, prompt, config, None)
}

pub fn generate_lexeme_softmax_with_priors(
    model: &LexemeSoftmaxModel,
    prompt: &[u16],
    config: LexemeGenerationConfig,
    decode_priors: Option<&LexemeDecodePriors>,
) -> Result<LexemeGenerationTrace, TrainError> {
    if prompt.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    validate_lexeme_decode_priors(model.vocab_size, config.decode, decode_priors)?;

    let d_model = model
        .embedding_dim
        .checked_add(1)
        .ok_or(TrainError::InvalidConfig)?;
    if model.seq_len == 0 || !model.seq_len.is_power_of_two() {
        return Err(TrainError::InvalidConfig);
    }
    let mut context = prompt.to_vec();
    if context
        .iter()
        .any(|&token| usize::from(token) >= model.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_window = lexeme_generation_context_window(&context, model.seq_len)?;
        let features = lexeme_context_features_q15_from_parts(
            &model.embeddings,
            model.vocab_size,
            model.embedding_dim,
            &context_window,
        )?;
        let row =
            lexeme_softmax_row_for(&model.output_weights, &features, model.vocab_size, d_model)?;
        let selection = select_lexeme_from_row(
            &row.logits_q8,
            &row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        );
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_tokens.push(predicted_token);
        context.push(predicted_token);
        steps.push(LexemeGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: row.logits_q8[predicted_index],
            predicted_probability_q15: row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });
    }

    Ok(LexemeGenerationTrace {
        config,
        prompt_tokens: prompt.to_vec(),
        generated_tokens,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        output_weight_hash: model.output_weight_hash(),
        context_seq_len: model.seq_len,
        decode_priors: decode_priors.map(LexemeDecodePriors::trace),
        steps,
    })
}

pub fn generate_mini_transformer(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_priors(model, prompt, config, None)
}

pub fn generate_mini_transformer_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
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
        let cache = mini_transformer_forward_for(model, context_window)?;
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
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
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
}

fn accumulate_linear_weight_gradient_i64_prescaled(
    input: &[i16],
    scaled_grad_output: &[i32],
    gradient: &mut LinearWeightGradientI64,
) -> Result<(), TrainError> {
    if input.len() != gradient.input_dim || scaled_grad_output.len() != gradient.output_dim {
        return Err(TrainError::InvalidConfig);
    }

    for (out_index, &scaled_grad) in scaled_grad_output.iter().enumerate() {
        if scaled_grad == 0 {
            continue;
        }
        let row_start = out_index
            .checked_mul(gradient.input_dim)
            .ok_or(TrainError::CoreRejected("linear_weight_gradient_row"))?;
        for (in_index, &activation) in input.iter().enumerate() {
            if activation == 0 {
                continue;
            }
            let product = i64::from(scaled_grad)
                .checked_mul(i64::from(activation))
                .ok_or(TrainError::CoreRejected("linear_weight_gradient_product"))?;
            let index = row_start
                .checked_add(in_index)
                .ok_or(TrainError::CoreRejected("linear_weight_gradient_index"))?;
            gradient.accumulators[index] =
                gradient.accumulators[index].checked_add(product).ok_or(
                    TrainError::CoreRejected("linear_weight_gradient_accumulate"),
                )?;
        }
    }

    gradient.sample_count =
        gradient
            .sample_count
            .checked_add(1)
            .ok_or(TrainError::CoreRejected(
                "linear_weight_gradient_sample_count",
            ))?;
    Ok(())
}

fn apply_linear_weight_gradient_i64_to_i8(
    gradient: &mut LinearWeightGradientI64,
    weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<LinearWeightUpdateStats, TrainError> {
    if weights.len() != gradient.accumulators.len()
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = empty_linear_weight_update_stats();
    if gradient.sample_count == 0 {
        return Ok(stats);
    }

    for ((raw_sum, residual), weight) in gradient
        .accumulators
        .iter()
        .zip(gradient.residuals.iter_mut())
        .zip(weights.iter_mut())
    {
        if *raw_sum == 0 {
            continue;
        }

        let averaged = round_div_i64(*raw_sum, gradient.sample_count)?;
        let product =
            averaged
                .checked_mul(i64::from(learning_rate))
                .ok_or(TrainError::CoreRejected(
                    "linear_weight_gradient_apply_product",
                ))?;
        let product = if carry_residual {
            product
                .checked_add(*residual)
                .ok_or(TrainError::CoreRejected(
                    "linear_weight_gradient_apply_residual",
                ))?
        } else {
            product
        };
        let scaled_update = round_shift_rhu_i64(product, learning_rate_shift);
        let next_residual = if carry_residual {
            rounded_shift_residual_i64(product, scaled_update, learning_rate_shift)?
        } else {
            0
        };
        let delta = -scaled_update;
        if delta == 0 {
            stats.zero_delta_count =
                stats
                    .zero_delta_count
                    .checked_add(1)
                    .ok_or(TrainError::CoreRejected(
                        "linear_weight_gradient_zero_delta",
                    ))?;
        }

        let previous = *weight;
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainError::CoreRejected(
                "linear_weight_gradient_apply_delta",
            ))?;
        let clamped = saturate_i8(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainError::CoreRejected(
                    "linear_weight_gradient_saturation",
                ))?;
            *residual = 0;
        } else {
            *residual = next_residual;
        }
        let applied_delta = i64::from(clamped) - i64::from(previous);
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .checked_add(applied_delta.unsigned_abs())
            .ok_or(TrainError::CoreRejected("linear_weight_gradient_delta_l1"))?;
        *weight = clamped;
    }

    gradient.clear();
    Ok(stats)
}

fn rounded_shift_residual_i64(
    value: i64,
    shifted: i64,
    right_shift: u8,
) -> Result<i64, TrainError> {
    if right_shift == 0 {
        return Ok(0);
    }

    let applied = (i128::from(shifted))
        .checked_shl(u32::from(right_shift))
        .ok_or(TrainError::CoreRejected(
            "linear_weight_gradient_residual_shift",
        ))?;
    let residual = i128::from(value)
        .checked_sub(applied)
        .ok_or(TrainError::CoreRejected(
            "linear_weight_gradient_residual_subtract",
        ))?;
    i64::try_from(residual)
        .map_err(|_| TrainError::CoreRejected("linear_weight_gradient_residual_overflow"))
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
}

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
) -> Result<GatedMlpWeightUpdateStats, TrainError> {
    Ok(GatedMlpWeightUpdateStats {
        down: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.down,
            down_weights,
            learning_rate,
            learning_rate_shift,
            false,
        )?,
        up: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.up,
            up_weights,
            learning_rate,
            learning_rate_shift,
            false,
        )?,
        gate: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.gate,
            gate_weights,
            learning_rate,
            learning_rate_shift,
            false,
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
}

fn accumulate_mini_transformer_attention_weight_gradient_i64(
    cache: &MiniTransformerMlpForwardCache,
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
    let q = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.q,
        &mut model.q_weights,
        config.learning_rate,
        config.attention_qk_learning_rate_shift,
        false,
    )?;
    let k = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.k,
        &mut model.k_weights,
        config.learning_rate,
        config.attention_qk_learning_rate_shift,
        false,
    )?;
    let (v, o) = if config.attention_vo_oracle {
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
                &mut model.v_weights,
                config.learning_rate,
                config.attention_learning_rate_shift,
                config.attention_vo_error_feedback,
            )?,
            apply_linear_weight_gradient_i64_to_i8(
                &mut gradient.o,
                &mut model.o_weights,
                config.learning_rate,
                config.attention_learning_rate_shift,
                config.attention_vo_error_feedback,
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
    if starts.is_empty() || seq_len == 0 || step <= 0 {
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
    let len = match matrix {
        MiniTransformerAttentionVoMatrix::Value => model.v_weights.len(),
        MiniTransformerAttentionVoMatrix::Output => model.o_weights.len(),
    };
    if len != MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidConfig);
    }

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
    match matrix {
        MiniTransformerAttentionVoMatrix::Value => model
            .v_weights
            .get(index)
            .copied()
            .ok_or(TrainError::InvalidConfig),
        MiniTransformerAttentionVoMatrix::Output => model
            .o_weights
            .get(index)
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
    let slot = match matrix {
        MiniTransformerAttentionVoMatrix::Value => model.v_weights.get_mut(index),
        MiniTransformerAttentionVoMatrix::Output => model.o_weights.get_mut(index),
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
}

impl MiniTransformerEmbeddingGradientI64 {
    fn new(context_seq_len: usize) -> Option<Self> {
        Some(Self {
            sample_count: 0,
            token_accumulators: vec![0_i64; BYTE_VOCAB.checked_mul(MINI_TRANSFORMER_D_MODEL)?],
            position_accumulators: vec![
                0_i64;
                context_seq_len.checked_mul(MINI_TRANSFORMER_D_MODEL)?
            ],
        })
    }

    fn clear(&mut self) {
        self.sample_count = 0;
        self.token_accumulators.fill(0);
        self.position_accumulators.fill(0);
    }
}

fn accumulate_mini_transformer_embedding_gradient_i64(
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    gradient: &mut MiniTransformerEmbeddingGradientI64,
) -> Result<(), TrainError> {
    if context.is_empty()
        || grad_embedding_output_q15.len()
            != context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
        || gradient.token_accumulators.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || gradient.position_accumulators.len()
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

    gradient.sample_count = gradient
        .sample_count
        .checked_add(1)
        .ok_or(TrainError::CoreRejected("embedding_gradient_sample_count"))?;
    Ok(())
}

fn apply_mini_transformer_embedding_gradient_i64_to_i16(
    gradient: &mut MiniTransformerEmbeddingGradientI64,
    embeddings: &mut [i16],
    position_embeddings: &mut [i16],
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != gradient.token_accumulators.len()
        || position_embeddings.len() != gradient.position_accumulators.len()
        || learning_rate <= 0
        || embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = empty_softmax_update_stats();
    if gradient.sample_count == 0 {
        return Ok(stats);
    }

    apply_embedding_accumulators_i64_to_i16(
        &gradient.token_accumulators,
        embeddings,
        gradient.sample_count,
        learning_rate,
        embedding_learning_rate_shift,
        &mut stats,
    )?;
    apply_embedding_accumulators_i64_to_i16(
        &gradient.position_accumulators,
        position_embeddings,
        gradient.sample_count,
        learning_rate,
        embedding_learning_rate_shift,
        &mut stats,
    )?;

    gradient.clear();
    Ok(stats)
}

fn apply_embedding_accumulators_i64_to_i16(
    accumulators: &[i64],
    embeddings: &mut [i16],
    sample_count: usize,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
    stats: &mut SoftmaxUpdateStats,
) -> Result<(), TrainError> {
    for (raw_sum, embedding) in accumulators.iter().zip(embeddings.iter_mut()) {
        if *raw_sum == 0 {
            continue;
        }
        let averaged = round_div_i64(*raw_sum, sample_count)?;
        let product = averaged
            .checked_mul(i64::from(learning_rate))
            .ok_or(TrainError::CoreRejected("embedding_gradient_apply_product"))?;
        let scaled_update = round_shift_rhu_i64(product, embedding_learning_rate_shift);
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

fn mini_transformer_validate_guard_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
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
        mini_transformer_forward_for(model, &tokens[start..start + seq_len])?;
    }

    Ok(())
}

fn mini_transformer_validate_batch_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
) -> Result<(), TrainError> {
    if starts.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for &start in starts {
        mini_transformer_forward_for(model, &tokens[start..start + seq_len])?;
    }

    Ok(())
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

fn initial_mini_transformer_position_embeddings(context_seq_len: usize) -> Vec<i16> {
    let mut embeddings = Vec::with_capacity(context_seq_len * MINI_TRANSFORMER_D_MODEL);
    for position in 0..context_seq_len {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            embeddings.push(mini_transformer_position_signal_q15(position, dim));
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
    position_embeddings: &[i16],
    context: &[u8],
) -> Result<Vec<i16>, TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || position_embeddings.len() < context.len() * MINI_TRANSFORMER_D_MODEL
        || context.is_empty()
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
        let position_row = position_embeddings
            .get(position_start..position_start + MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel(
                "mini transformer position embedding row",
            ))?;
        for (&value, &position_value) in row.iter().zip(position_row.iter()) {
            output.push(saturate_i16(i64::from(value) + i64::from(position_value)));
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
    let embedding_output = mini_transformer_embedding_sequence_q15(
        &model.embeddings,
        &model.position_embeddings,
        context,
    )?;
    let attention_norm = embedding_output.clone();

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
    let attention_probabilities_q15 =
        mini_transformer_attention_probabilities_q15(seq_len, &q, &k)?;

    let mut residual_saturation_count = 0_usize;
    let mut attention_residual = vec![0_i16; total];
    residual_saturation_count += add_i16_residual_rows_checked(
        &embedding_output,
        &attention_output,
        &mut attention_residual,
    )?;
    let mlp_norm = attention_residual.clone();

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

    let last_start = (seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
    let row = mini_transformer_output_row_for(
        &model.output_weights,
        &block_output[last_start..last_end],
    )?;

    Ok(MiniTransformerMlpForwardCache {
        embedding_output,
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
    let mut grad_embedding_output = vec![0_i16; total];
    let mut input_gradient_saturation_count = 0_usize;

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        let mut grad_q_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_k_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_v_input = [0_i16; MINI_TRANSFORMER_D_MODEL];

        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_q[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.q_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
            &mut grad_q_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_q_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_k[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.k_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad,
            },
            &mut grad_k_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_k_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_v[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.v_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad,
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
            grad_embedding_output[row_start + dim] = saturate_i16(scaled);
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
            grad_attention_output,
            &grad_q,
            &grad_k,
            &grad_v,
            attention_gradient,
            &mut scaled_grad,
        )?;
    } else {
        for token in 0..seq_len {
            let row_start = token * MINI_TRANSFORMER_D_MODEL;
            let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
            let q_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
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
                &cache.attention_norm[row_start..row_end],
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
                &cache.attention_norm[row_start..row_end],
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
        grad_embedding_output,
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
    Ok(mini_transformer_eval_counts_strict(tokens, starts, model, seq_len)?.mistakes)
}

fn mini_transformer_total_probability_error_q15(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<usize, TrainError> {
    Ok(mini_transformer_eval_counts_strict(tokens, starts, model, seq_len)?.probability_error_q15)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerEvalSummary {
    mistakes: usize,
    probability_error_q15: usize,
    invalid_forward_count: usize,
    logits_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerEvalCounts {
    mistakes: usize,
    probability_error_q15: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerWindowEvalRecord {
    start: usize,
    end: usize,
    mistakes: usize,
    probability_error_q15: usize,
    invalid_forward_count: usize,
    logits_q8: Option<[i32; BYTE_VOCAB]>,
}

fn mini_transformer_eval_counts_strict(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
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

            let cache = mini_transformer_forward_for(model, &tokens[start..end])?;
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

fn mini_transformer_eval_summary(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<MiniTransformerEvalSummary, TrainError> {
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

            let record = match mini_transformer_forward_for(model, &tokens[start..end]) {
                Ok(cache) => {
                    let mistakes = usize::from(byte_argmax_i32(&cache.logits_q8) != tokens[end]);
                    let probability_error_q15 =
                        byte_sample_probability_error_q15(&cache.probabilities_q15, tokens[end]);
                    MiniTransformerWindowEvalRecord {
                        start,
                        end,
                        mistakes,
                        probability_error_q15,
                        invalid_forward_count: 0,
                        logits_q8: Some(cache.logits_q8),
                    }
                }
                Err(_) => MiniTransformerWindowEvalRecord {
                    start,
                    end,
                    mistakes: 1,
                    probability_error_q15: i16::MAX as usize,
                    invalid_forward_count: 1,
                    logits_q8: None,
                },
            };
            records.push(record);
        }
        Ok(records)
    })?;

    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    let mut invalid_forward_count = 0_usize;
    let mut hasher = StableHasher::new();

    for record in chunks.into_iter().flatten() {
        mistakes = mistakes.saturating_add(record.mistakes);
        probability_error_q15 = probability_error_q15.saturating_add(record.probability_error_q15);
        invalid_forward_count = invalid_forward_count.saturating_add(record.invalid_forward_count);
        hasher.update_usize(record.start);
        if let Some(logits_q8) = record.logits_q8 {
            hasher.update_i32_slice(&logits_q8);
        } else {
            hasher.update_u8(0xff);
            hasher.update_bytes(&tokens[record.start..=record.end]);
        }
    }

    Ok(MiniTransformerEvalSummary {
        mistakes,
        probability_error_q15,
        invalid_forward_count,
        logits_hash: hasher.finish(),
    })
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

fn lexeme_total_error(
    tokens: &[u16],
    starts: &[usize],
    embedding_model: &LexemeEmbeddingModel,
    output_weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let d_model = embedding_model
        .embedding_dim
        .checked_add(1)
        .ok_or(TrainError::InvalidConfig)?;
    let mut mistakes = 0_usize;
    for &start in starts {
        let features =
            lexeme_context_features_q15(embedding_model, &tokens[start..start + seq_len])?;
        let row = lexeme_softmax_row_for(
            output_weights,
            &features,
            embedding_model.vocab_size,
            d_model,
        )?;
        if lexeme_argmax_i32(&row.logits_q8) != tokens[start + seq_len] {
            mistakes = mistakes.saturating_add(1);
        }
    }
    Ok(mistakes)
}

fn lexeme_total_probability_error_q15(
    tokens: &[u16],
    starts: &[usize],
    embedding_model: &LexemeEmbeddingModel,
    output_weights: &[i8],
    seq_len: usize,
) -> Result<usize, TrainError> {
    let d_model = embedding_model
        .embedding_dim
        .checked_add(1)
        .ok_or(TrainError::InvalidConfig)?;
    let mut error = 0_usize;
    for &start in starts {
        let features =
            lexeme_context_features_q15(embedding_model, &tokens[start..start + seq_len])?;
        let row = lexeme_softmax_row_for(
            output_weights,
            &features,
            embedding_model.vocab_size,
            d_model,
        )?;
        error = error.saturating_add(lexeme_sample_probability_error_q15(
            &row.probabilities_q15,
            tokens[start + seq_len],
        ));
    }
    Ok(error)
}

fn lexeme_sample_probability_error_q15(probabilities_q15: &[i16], target: u16) -> usize {
    let target = usize::from(target);
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities_q15[target])).max(0) as usize;
    for (class_id, &probability) in probabilities_q15.iter().enumerate() {
        if class_id != target {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
}

fn lexeme_softmax_learning_rate_shift_for_update(
    config: LexemeSoftmaxTrainConfig,
    update_index: usize,
) -> u8 {
    if config.lr_shift_decay_windows == 0 {
        return config.learning_rate_shift;
    }
    let phase = update_index / config.lr_shift_decay_windows;
    let phase_shift = phase
        .saturating_mul(usize::from(config.lr_shift_decay_step))
        .min(usize::from(MAX_RIGHT_SHIFT));
    let shifted = usize::from(config.learning_rate_shift)
        .saturating_add(phase_shift)
        .min(usize::from(config.max_learning_rate_shift));
    shifted as u8
}

fn valid_q15_weight_floor(value: i16) -> bool {
    value > 0
}

fn lexeme_frequency_weights_q15(
    tokens: &[u16],
    vocab_size: usize,
    frequency_cap: u32,
    min_weight_q15: i16,
) -> Result<Vec<i16>, TrainError> {
    if vocab_size == 0
        || tokens.iter().any(|&token| usize::from(token) >= vocab_size)
        || !valid_q15_weight_floor(min_weight_q15)
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut weights = vec![i16::MAX; vocab_size];
    if frequency_cap == 0 {
        return Ok(weights);
    }

    let mut counts = vec![0_u32; vocab_size];
    for &token in tokens {
        let count = &mut counts[usize::from(token)];
        *count = count.saturating_add(1);
    }
    for (index, &count) in counts.iter().enumerate() {
        weights[index] = lexeme_frequency_weight_q15(count, frequency_cap, min_weight_q15);
    }
    Ok(weights)
}

fn lexeme_frequency_weight_q15(count: u32, frequency_cap: u32, min_weight_q15: i16) -> i16 {
    if frequency_cap == 0 || count == 0 || count <= frequency_cap {
        return i16::MAX;
    }

    let ratio_q30 = (u64::from(frequency_cap) << (Q15_SHIFT * 2)) / u64::from(count);
    let weight = integer_sqrt_u64(ratio_q30).min(i16::MAX as u64) as i16;
    weight.max(min_weight_q15)
}

fn lexeme_pair_frequency_weight_q15(left_token: u16, right_token: u16, weights: &[i16]) -> i16 {
    weights[usize::from(left_token)].min(weights[usize::from(right_token)])
}

fn lexeme_training_quality_weights_q15(
    profile: LexemeQualityWeightProfile,
    weights: Option<&[i16]>,
    vocab_size: usize,
) -> Result<Vec<i16>, TrainError> {
    match profile {
        LexemeQualityWeightProfile::Off => Ok(vec![i16::MAX; vocab_size]),
        LexemeQualityWeightProfile::CruftAware => {
            let weights = weights.ok_or(TrainError::InvalidConfig)?;
            if weights.len() != vocab_size
                || weights
                    .iter()
                    .any(|&weight| !valid_q15_weight_floor(weight))
            {
                return Err(TrainError::InvalidConfig);
            }
            Ok(weights.to_vec())
        }
    }
}

pub fn lexeme_quality_weights_from_vocab(
    vocab: &[String],
    vocab_size: usize,
    profile: LexemeQualityWeightProfile,
) -> Result<Vec<i16>, TrainError> {
    if vocab_size == 0 || vocab.len() < vocab_size {
        return Err(TrainError::InvalidConfig);
    }
    let mut weights = vec![i16::MAX; vocab_size];
    if profile == LexemeQualityWeightProfile::Off {
        return Ok(weights);
    }
    for (index, lexeme) in vocab.iter().take(vocab_size).enumerate() {
        weights[index] = lexeme_quality_weight_q15(lexeme);
    }
    Ok(weights)
}

fn lexeme_quality_weight_q15(lexeme: &str) -> i16 {
    let lower = lexeme.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return 4096;
    }
    if is_cruft_lexeme(&lower) {
        return 4096;
    }
    if lower.starts_with("http") || lower == "www" || lower.ends_with("html") {
        return 4096;
    }
    if lower.len() > 24 && lower.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return 8192;
    }
    i16::MAX
}

fn is_cruft_lexeme(lexeme: &str) -> bool {
    matches!(
        lexeme,
        "align"
            | "background"
            | "bgcolor"
            | "border"
            | "caption"
            | "category"
            | "center"
            | "cite"
            | "class"
            | "color"
            | "colspan"
            | "copyright"
            | "div"
            | "ebook"
            | "file"
            | "font"
            | "gutenberg"
            | "height"
            | "html"
            | "http"
            | "https"
            | "image"
            | "isbn"
            | "license"
            | "px"
            | "ref"
            | "rowspan"
            | "small"
            | "span"
            | "style"
            | "table"
            | "template"
            | "thumb"
            | "url"
            | "width"
            | "www"
    )
}

fn lexeme_combine_q15_weights(left: i16, right: i16) -> i16 {
    if left == i16::MAX {
        return right;
    }
    if right == i16::MAX {
        return left;
    }
    round_shift_rhu_i64(i64::from(left).saturating_mul(i64::from(right)), Q15_SHIFT)
        .clamp(1, i64::from(i16::MAX)) as i16
}

fn lexeme_scale_gradient_q15(gradient_q15: &[i32], frequency_weight_q15: i16) -> Vec<i32> {
    if frequency_weight_q15 == i16::MAX {
        return gradient_q15.to_vec();
    }
    gradient_q15
        .iter()
        .map(|&gradient| {
            round_shift_rhu_i64(
                i64::from(gradient).saturating_mul(i64::from(frequency_weight_q15)),
                Q15_SHIFT,
            ) as i32
        })
        .collect()
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

fn hash_lexeme_logits(
    tokens: &[u16],
    starts: &[usize],
    embedding_model: &LexemeEmbeddingModel,
    output_weights: &[i8],
    seq_len: usize,
) -> Result<u64, TrainError> {
    let d_model = embedding_model
        .embedding_dim
        .checked_add(1)
        .ok_or(TrainError::InvalidConfig)?;
    let mut hasher = StableHasher::new();
    for &start in starts {
        let features =
            lexeme_context_features_q15(embedding_model, &tokens[start..start + seq_len])?;
        let row = lexeme_softmax_row_for(
            output_weights,
            &features,
            embedding_model.vocab_size,
            d_model,
        )?;
        hasher.update_i32_slice(&row.logits_q8);
    }
    Ok(hasher.finish())
}

fn hash_byte_windows(tokens: &[u8], config: ByteSoftmaxTrainConfig, starts: &[usize]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.seq_len);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.window_offset);
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
    hasher.update_usize(config.window_offset);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(config.batch_windows);
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
    hasher.update_usize(config.window_offset);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(BYTE_EMBED_DIM);
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_bytes(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
}

fn hash_lexeme_windows(
    tokens: &[u16],
    config: LexemeEmbeddingTrainConfig,
    starts: &[usize],
) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.context_radius);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.window_offset);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(config.vocab_size);
    hasher.update_usize(config.embedding_dim);
    hasher.update_usize(config.concept_frequency_cap as usize);
    hasher.update_usize(config.concept_frequency_min_weight_q15 as usize);
    hasher.update_usize(lexeme_quality_weight_profile_id(
        config.quality_weight_profile,
    ));
    for &center in starts {
        hasher.update_usize(center);
        let start = center - config.context_radius;
        let end = center + config.context_radius + 1;
        hasher.update_u16_slice(&tokens[start..end]);
    }
    hasher.finish()
}

fn hash_lexeme_softmax_windows(
    tokens: &[u16],
    config: LexemeSoftmaxTrainConfig,
    starts: &[usize],
) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.seq_len);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.window_offset);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(config.target_frequency_cap as usize);
    hasher.update_usize(config.target_frequency_min_weight_q15 as usize);
    hasher.update_usize(lexeme_quality_weight_profile_id(
        config.quality_weight_profile,
    ));
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_u16_slice(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
}

fn decode_u16_tokens(bytes: &[u8]) -> Result<Vec<u16>, TrainError> {
    if bytes.is_empty() || bytes.len() % 2 != 0 {
        return Err(TrainError::InvalidConfig);
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn lexeme_center_starts(tokens_len: usize, config: LexemeEmbeddingTrainConfig) -> Vec<usize> {
    let mut starts = Vec::new();
    let Some(first_center) = config.context_radius.checked_add(config.window_offset) else {
        return starts;
    };
    let Some(last_exclusive) = tokens_len.checked_sub(config.context_radius) else {
        return starts;
    };
    let mut center = first_center;
    while center < last_exclusive {
        if config
            .max_windows
            .is_some_and(|limit| starts.len() >= limit)
        {
            break;
        }
        starts.push(center);
        center = center.saturating_add(config.stride);
    }
    starts
}

fn lexeme_softmax_starts(
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

fn initial_lexeme_embeddings(
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<Vec<i16>, TrainError> {
    let total = vocab_size
        .checked_mul(embedding_dim)
        .ok_or(TrainError::InvalidConfig)?;
    let mut embeddings = Vec::with_capacity(total);
    for token in 0..vocab_size {
        for dim in 0..embedding_dim {
            let mut hasher = StableHasher::new();
            hasher.update_usize(token);
            hasher.update_usize(dim);
            let raw = (hasher.finish() & 0x03ff) as i32;
            embeddings.push((raw - 512) as i16);
        }
    }
    Ok(embeddings)
}

fn lexeme_context_features_q15(
    model: &LexemeEmbeddingModel,
    context: &[u16],
) -> Result<Vec<i16>, TrainError> {
    lexeme_context_features_q15_from_parts(
        &model.embeddings,
        model.vocab_size,
        model.embedding_dim,
        context,
    )
}

fn lexeme_context_features_q15_from_parts(
    embeddings: &[i16],
    vocab_size: usize,
    embedding_dim: usize,
    context: &[u16],
) -> Result<Vec<i16>, TrainError> {
    if context.is_empty()
        || !context.len().is_power_of_two()
        || embeddings.len()
            != vocab_size
                .checked_mul(embedding_dim)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }
    let shift = context.len().trailing_zeros() as u8;
    let mut features = Vec::with_capacity(embedding_dim + 1);
    features.push(i16::MAX);
    for dim in 0..embedding_dim {
        let mut acc = 0_i64;
        for &token in context {
            let token = usize::from(token);
            if token >= vocab_size {
                return Err(TrainError::InvalidConfig);
            }
            acc = acc.saturating_add(i64::from(embeddings[token * embedding_dim + dim]));
        }
        features.push(saturate_i16(round_shift_rhu_i64(acc, shift)));
    }
    Ok(features)
}

fn lexeme_generation_context_window(
    context: &[u16],
    seq_len: usize,
) -> Result<Vec<u16>, TrainError> {
    if context.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    if context.len() >= seq_len {
        Ok(context[context.len() - seq_len..].to_vec())
    } else {
        let pad = context[0];
        let mut window = Vec::with_capacity(seq_len);
        window.resize(seq_len - context.len(), pad);
        window.extend_from_slice(context);
        Ok(window)
    }
}

fn lexeme_pair_dot_i64(
    embeddings: &[i16],
    embedding_dim: usize,
    left_token: u16,
    right_token: u16,
) -> i64 {
    let left_start = usize::from(left_token) * embedding_dim;
    let right_start = usize::from(right_token) * embedding_dim;
    let mut acc = 0_i64;
    for dim in 0..embedding_dim {
        acc = acc.saturating_add(
            i64::from(embeddings[left_start + dim]) * i64::from(embeddings[right_start + dim]),
        );
    }
    acc
}

fn lexeme_total_positive_dot_i64(
    tokens: &[u16],
    starts: &[usize],
    embeddings: &[i16],
    config: LexemeEmbeddingTrainConfig,
) -> i64 {
    let mut total = 0_i64;
    for &center in starts {
        let center_token = tokens[center];
        for context in center - config.context_radius..=center + config.context_radius {
            if context == center {
                continue;
            }
            total = total.saturating_add(lexeme_pair_dot_i64(
                embeddings,
                config.embedding_dim,
                center_token,
                tokens[context],
            ));
        }
    }
    total
}

fn lexeme_total_negative_dot_i64(
    tokens: &[u16],
    starts: &[usize],
    embeddings: &[i16],
    config: LexemeEmbeddingTrainConfig,
) -> i64 {
    let mut total = 0_i64;
    let mut update_index = 0_usize;
    for &center in starts {
        let center_token = tokens[center];
        for context in center - config.context_radius..=center + config.context_radius {
            if context == center {
                continue;
            }
            let context_token = tokens[context];
            let negative_token =
                lexeme_negative_token(center_token, context_token, update_index, config.vocab_size);
            total = total.saturating_add(lexeme_pair_dot_i64(
                embeddings,
                config.embedding_dim,
                center_token,
                negative_token,
            ));
            update_index = update_index.saturating_add(1);
        }
    }
    total
}

fn lexeme_negative_token(
    center_token: u16,
    context_token: u16,
    update_index: usize,
    vocab_size: usize,
) -> u16 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(usize::from(center_token));
    hasher.update_usize(usize::from(context_token));
    hasher.update_usize(update_index);
    let mut candidate = (hasher.finish() % vocab_size as u64) as usize;
    if candidate == usize::from(center_token) || candidate == usize::from(context_token) {
        candidate = (candidate + 1) % vocab_size;
    }
    if candidate == usize::from(center_token) || candidate == usize::from(context_token) {
        candidate = (candidate + 1) % vocab_size;
    }
    candidate as u16
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct LexemeSoftmaxRow {
    logits_q8: Vec<i32>,
    probabilities_q15: Vec<i16>,
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

fn lexeme_softmax_row_for(
    weights: &[i8],
    features: &[i16],
    vocab_size: usize,
    d_model: usize,
) -> Result<LexemeSoftmaxRow, TrainError> {
    if vocab_size == 0
        || d_model == 0
        || features.len() != d_model
        || weights.len()
            != vocab_size
                .checked_mul(d_model)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut logits_q8 = vec![0_i32; vocab_size];
    for (class_id, out) in logits_q8.iter_mut().enumerate() {
        let row_start = class_id * d_model;
        let mut acc = 0_i64;
        for feature_index in 0..d_model {
            acc = acc.saturating_add(
                i64::from(features[feature_index]) * i64::from(weights[row_start + feature_index]),
            );
        }
        *out = saturate_i16(round_shift_rhu_i64(acc, LEXEME_LOGIT_RIGHT_SHIFT)) as i32;
    }

    let mut probabilities_q15 = vec![0_i16; vocab_size];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("lexeme_output_head_softmax"))?;

    Ok(LexemeSoftmaxRow {
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

fn lexeme_softmax_gradient_q15(probabilities_q15: &[i16], target: u16) -> Vec<i32> {
    let target = usize::from(target);
    let mut gradient = vec![0_i32; probabilities_q15.len()];
    for (class_id, out) in gradient.iter_mut().enumerate() {
        *out = i32::from(probabilities_q15[class_id]);
        if class_id == target {
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

fn lexeme_argmax_i32(logits: &[i32]) -> u16 {
    logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, core::cmp::Reverse(index)))
        .map(|(index, _)| index as u16)
        .unwrap_or(0)
}

fn lexeme_argmax_concept_i32(logits: &[i32]) -> u16 {
    logits
        .iter()
        .enumerate()
        .skip(256)
        .max_by_key(|&(index, &logit)| (logit, core::cmp::Reverse(index)))
        .map(|(index, _)| index as u16)
        .unwrap_or_else(|| lexeme_argmax_i32(logits))
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
    if decode.corpus_prior {
        if let (Some(priors), Some(&previous)) = (decode_priors, context.last()) {
            let prior_q15 = priors.transition_probability_q15(previous, candidate as u8);
            let bonus = (weight.saturating_mul(u64::from(prior_q15))) >> Q15_SHIFT;
            weight = weight.saturating_add(bonus);
        }
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
        || decode.corpus_prior
        || decode.strict_adjacency
        || (decode.repeat_window > 0 && decode.repeat_penalty_shift > 0)
}

fn validate_decode_priors(
    decode: DecodeConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<(), TrainError> {
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
    if decode.corpus_prior {
        if let (Some(priors), Some(&previous)) = (decode_priors, context.last()) {
            let prior_q15 = i32::from(priors.transition_probability_q15(previous, candidate as u8));
            let shift = decode.corpus_prior_logit_shift.min(30);
            logit = logit.saturating_add(prior_q15 >> shift);
        }
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

fn decode_sample_u64(seed: u64, step_index: usize, context: &[u8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_bytes(&seed.to_le_bytes());
    hasher.update_usize(step_index);
    hasher.update_u8_slice(context);
    splitmix64(hasher.finish())
}

fn select_lexeme_from_row(
    logits_q8: &[i32],
    probabilities_q15: &[i16],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> LexemeDecodeSelection {
    match decode.strategy {
        DecodeStrategy::Greedy => lexeme_decode_fallback_selection(
            logits_q8,
            probabilities_q15,
            decode,
            context,
            decode_priors,
        ),
        DecodeStrategy::Sample => lexeme_sample_from_probabilities_q15(
            logits_q8,
            probabilities_q15,
            decode,
            step_index,
            context,
            decode_priors,
        )
        .unwrap_or_else(|| {
            lexeme_decode_fallback_selection(
                logits_q8,
                probabilities_q15,
                decode,
                context,
                decode_priors,
            )
        }),
    }
}

fn lexeme_sample_from_probabilities_q15(
    logits_q8: &[i32],
    probabilities_q15: &[i16],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> Option<LexemeDecodeSelection> {
    let candidate_set = lexeme_decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let rejected_candidates = candidate_set.rejected_candidates;
    let mut mass = 0_u64;
    for &candidate in candidates.iter() {
        mass = mass.saturating_add(lexeme_decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        ));
    }
    if mass == 0 {
        return None;
    }

    let mut threshold = lexeme_decode_sample_u64(decode.sample_seed, step_index, context) % mass;
    for &candidate in candidates.iter() {
        let weight = lexeme_decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        );
        if threshold < weight {
            return Some(LexemeDecodeSelection {
                token: candidate as u16,
                candidate_count: candidates.len(),
                rejected_candidates,
            });
        }
        threshold -= weight;
    }
    None
}

fn lexeme_decode_fallback_selection(
    logits_q8: &[i32],
    probabilities_q15: &[i16],
    decode: DecodeConfig,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> LexemeDecodeSelection {
    let candidate_set = lexeme_decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let token = if decode.repeat_window == 0
        && decode.repeat_penalty_shift == 0
        && !decode.corpus_prior
    {
        candidates
            .first()
            .copied()
            .unwrap_or_else(|| usize::from(lexeme_argmax_concept_i32(logits_q8))) as u16
    } else {
        candidates
            .iter()
            .copied()
            .max_by_key(|&candidate| {
                (
                    lexeme_decode_candidate_weight_q15(
                        probabilities_q15,
                        candidate,
                        decode,
                        context,
                        decode_priors,
                    ),
                    lexeme_decode_effective_logit_q8(
                        logits_q8,
                        candidate,
                        decode,
                        context,
                        decode_priors,
                    ),
                    core::cmp::Reverse(candidate),
                )
            })
            .unwrap_or_else(|| usize::from(lexeme_argmax_concept_i32(logits_q8))) as u16
    };
    LexemeDecodeSelection {
        token,
        candidate_count: candidates.len(),
        rejected_candidates: candidate_set.rejected_candidates,
    }
}

fn lexeme_decode_candidates(
    logits_q8: &[i32],
    decode: DecodeConfig,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> DecodeCandidateSet {
    let vocab_size = logits_q8.len();
    let top_k = if decode.top_k == 0 || decode.top_k > vocab_size {
        vocab_size
    } else {
        decode.top_k
    };
    let mut rejected_candidates = DecodeRejectStats::default();
    let mut candidates = Vec::with_capacity(vocab_size.saturating_sub(256));
    for candidate in 0..vocab_size {
        if candidate < 256 {
            rejected_candidates.byte_fallback += 1;
            continue;
        }
        let token = candidate as u16;
        if decode.max_repeat_run > 0
            && lexeme_would_exceed_repeat_run(token, context, decode.max_repeat_run)
        {
            rejected_candidates.repeat_run += 1;
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
            compare_lexeme_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
        });
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|&left, &right| {
        compare_lexeme_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
    });
    if candidates.is_empty() {
        candidates.push(usize::from(lexeme_argmax_concept_i32(logits_q8)));
    }
    DecodeCandidateSet {
        candidates,
        rejected_candidates,
    }
}

fn compare_lexeme_decode_candidates(
    left: usize,
    right: usize,
    logits_q8: &[i32],
    decode: DecodeConfig,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> core::cmp::Ordering {
    lexeme_decode_effective_logit_q8(logits_q8, right, decode, context, decode_priors)
        .cmp(&lexeme_decode_effective_logit_q8(
            logits_q8,
            left,
            decode,
            context,
            decode_priors,
        ))
        .then_with(|| left.cmp(&right))
}

fn lexeme_decode_candidate_weight_q15(
    probabilities_q15: &[i16],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> u64 {
    let mut weight = i32::from(probabilities_q15[candidate]).max(0) as u64;
    if decode.corpus_prior {
        if let (Some(priors), Some(&previous)) = (decode_priors, context.last()) {
            let prior_q15 = priors.transition_probability_q15(previous, candidate as u16);
            let bonus = (weight.saturating_mul(u64::from(prior_q15))) >> Q15_SHIFT;
            weight = weight.saturating_add(bonus);
        }
    }
    if decode.repeat_window > 0 && decode.repeat_penalty_shift > 0 {
        let repeat_count = recent_lexeme_count(candidate as u16, context, decode.repeat_window);
        let penalty_shift = repeat_count
            .saturating_mul(usize::from(decode.repeat_penalty_shift))
            .min(63);
        weight >>= penalty_shift;
    }
    weight
}

fn validate_lexeme_decode_priors(
    vocab_size: usize,
    decode: DecodeConfig,
    decode_priors: Option<&LexemeDecodePriors>,
) -> Result<(), TrainError> {
    if (decode.corpus_prior || decode.strict_adjacency) && decode_priors.is_none() {
        return Err(TrainError::InvalidConfig);
    }
    if let Some(priors) = decode_priors {
        if priors.vocab_size != vocab_size {
            return Err(TrainError::InvalidConfig);
        }
    }
    Ok(())
}

fn lexeme_decode_effective_logit_q8(
    logits_q8: &[i32],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u16],
    decode_priors: Option<&LexemeDecodePriors>,
) -> i32 {
    let mut logit = logits_q8[candidate];
    if decode.corpus_prior {
        if let (Some(priors), Some(&previous)) = (decode_priors, context.last()) {
            let prior_q15 =
                i32::from(priors.transition_probability_q15(previous, candidate as u16));
            let shift = decode.corpus_prior_logit_shift.min(30);
            logit = logit.saturating_add(prior_q15 >> shift);
        }
    }
    logit
}

fn recent_lexeme_count(candidate: u16, context: &[u16], repeat_window: usize) -> usize {
    context
        .iter()
        .rev()
        .take(repeat_window)
        .filter(|&&token| token == candidate)
        .count()
}

fn lexeme_would_exceed_repeat_run(candidate: u16, context: &[u16], max_repeat_run: usize) -> bool {
    let run_len = context
        .iter()
        .rev()
        .take_while(|&&token| token == candidate)
        .count();
    run_len >= max_repeat_run
}

fn lexeme_decode_sample_u64(seed: u64, step_index: usize, context: &[u16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_bytes(&seed.to_le_bytes());
    hasher.update_usize(step_index);
    hasher.update_u16_slice(context);
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

fn apply_lexeme_softmax_output_head_update(
    weights: &mut [i8],
    features: &[i16],
    gradient_q15: &[i32],
    vocab_size: usize,
    d_model: usize,
    learning_rate: i32,
    learning_rate_shift: u8,
    max_weight_delta: i32,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if vocab_size == 0
        || d_model == 0
        || features.len() != d_model
        || gradient_q15.len() != vocab_size
        || weights.len()
            != vocab_size
                .checked_mul(d_model)
                .ok_or(TrainError::InvalidConfig)?
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
        || max_weight_delta <= 0
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut gradient_saturation_count = 0_usize;
    let mut zero_delta_count = 0_usize;
    let mut weight_delta_l1 = 0_u64;
    let max_delta = i64::from(max_weight_delta);

    for (class_id, &gradient) in gradient_q15.iter().enumerate() {
        let row_start = class_id * d_model;
        for (feature_index, &activation) in features.iter().enumerate() {
            if activation == 0 || gradient == 0 {
                continue;
            }

            let product = i64::from(gradient)
                .saturating_mul(i64::from(activation))
                .saturating_mul(i64::from(learning_rate));
            let scaled_gradient = round_shift_rhu_i64(product, learning_rate_shift);
            let delta = (-scaled_gradient).clamp(-max_delta, max_delta);
            if delta == 0 {
                zero_delta_count = zero_delta_count.saturating_add(1);
            }

            let weight = &mut weights[row_start + feature_index];
            let previous = *weight;
            let unclamped = i64::from(previous).saturating_add(delta);
            let clamped = saturate_i8(unclamped);
            if i64::from(clamped) != unclamped {
                gradient_saturation_count = gradient_saturation_count.saturating_add(1);
            }
            let applied_delta = i64::from(clamped) - i64::from(previous);
            weight_delta_l1 = weight_delta_l1.saturating_add(applied_delta.unsigned_abs());
            *weight = clamped;
        }
    }

    Ok(SoftmaxUpdateStats {
        gradient_saturation_count,
        zero_delta_count,
        weight_delta_l1,
    })
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

fn apply_mini_transformer_embedding_update(
    embeddings: &mut [i16],
    position_embeddings: &mut [i16],
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || position_embeddings.len()
            < context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
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
            apply_embedding_delta_i16(
                &mut position_embeddings[position_row_start + dim],
                delta,
                &mut stats,
            );
        }
    }

    Ok(stats)
}

fn apply_lexeme_embedding_pair_update(
    embeddings: &mut [i16],
    embedding_dim: usize,
    left_token: u16,
    right_token: u16,
    direction: i32,
    learning_rate: i32,
    learning_rate_shift: u8,
    frequency_weight_q15: i16,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embedding_dim == 0
        || direction == 0
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
        || !valid_q15_weight_floor(frequency_weight_q15)
    {
        return Err(TrainError::InvalidConfig);
    }
    let total_shift = learning_rate_shift
        .checked_add(Q15_SHIFT)
        .ok_or(TrainError::InvalidConfig)?;
    if total_shift > MAX_RIGHT_SHIFT {
        return Err(TrainError::InvalidConfig);
    }
    let left_start = usize::from(left_token)
        .checked_mul(embedding_dim)
        .ok_or(TrainError::InvalidConfig)?;
    let right_start = usize::from(right_token)
        .checked_mul(embedding_dim)
        .ok_or(TrainError::InvalidConfig)?;
    if left_start + embedding_dim > embeddings.len()
        || right_start + embedding_dim > embeddings.len()
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    };

    for dim in 0..embedding_dim {
        let left_index = left_start + dim;
        let right_index = right_start + dim;
        let left_before = embeddings[left_index];
        let right_before = embeddings[right_index];
        let left_product = i64::from(right_before)
            .saturating_mul(i64::from(direction))
            .saturating_mul(i64::from(learning_rate))
            .saturating_mul(i64::from(frequency_weight_q15));
        let right_product = i64::from(left_before)
            .saturating_mul(i64::from(direction))
            .saturating_mul(i64::from(learning_rate))
            .saturating_mul(i64::from(frequency_weight_q15));
        let left_delta = round_shift_rhu_i64(left_product, total_shift);
        let right_delta = round_shift_rhu_i64(right_product, total_shift);
        if left_delta == 0 {
            stats.zero_delta_count = stats.zero_delta_count.saturating_add(1);
        }
        if right_delta == 0 {
            stats.zero_delta_count = stats.zero_delta_count.saturating_add(1);
        }
        apply_embedding_delta_i16(&mut embeddings[left_index], left_delta, &mut stats);
        apply_embedding_delta_i16(&mut embeddings[right_index], right_delta, &mut stats);
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

fn hash_u16_slice(values: &[u16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_u16_slice(values);
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

    fn update_u16_slice(&mut self, values: &[u16]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_bytes(&value.to_le_bytes());
        }
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

fn push_lexeme_embedding_steps_field(
    out: &mut String,
    name: &str,
    steps: &[LexemeEmbeddingTrainingStepTrace],
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
        push_usize_field(out, "center_index", step.center_index);
        comma(out);
        push_usize_field(out, "center_token", usize::from(step.center_token));
        comma(out);
        push_usize_field(out, "context_token", usize::from(step.context_token));
        comma(out);
        push_usize_field(out, "negative_token", usize::from(step.negative_token));
        comma(out);
        push_i16_field(
            out,
            "positive_frequency_weight_q15",
            step.positive_frequency_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "negative_frequency_weight_q15",
            step.negative_frequency_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "positive_quality_weight_q15",
            step.positive_quality_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "negative_quality_weight_q15",
            step.negative_quality_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "positive_update_weight_q15",
            step.positive_update_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "negative_update_weight_q15",
            step.negative_update_weight_q15,
        );
        comma(out);
        push_i64_field(out, "positive_dot_before_i64", step.positive_dot_before_i64);
        comma(out);
        push_i64_field(out, "positive_dot_after_i64", step.positive_dot_after_i64);
        comma(out);
        push_i64_field(out, "negative_dot_before_i64", step.negative_dot_before_i64);
        comma(out);
        push_i64_field(out, "negative_dot_after_i64", step.negative_dot_after_i64);
        comma(out);
        push_hash_field(out, "embedding_hash_before", step.embedding_hash_before);
        comma(out);
        push_hash_field(out, "embedding_hash_after", step.embedding_hash_after);
        comma(out);
        push_usize_field(out, "saturation_count", step.saturation_count);
        comma(out);
        push_usize_field(out, "zero_delta_count", step.zero_delta_count);
        comma(out);
        push_u64_field(out, "embedding_delta_l1", step.embedding_delta_l1);
        out.push('}');
    }
    out.push(']');
}

fn push_lexeme_softmax_steps_field(
    out: &mut String,
    name: &str,
    steps: &[LexemeSoftmaxTrainingStepTrace],
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
        push_usize_field(out, "previous_token", usize::from(step.previous_token));
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
        push_i16_field(
            out,
            "target_frequency_weight_q15",
            step.target_frequency_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "target_quality_weight_q15",
            step.target_quality_weight_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "target_update_weight_q15",
            step.target_update_weight_q15,
        );
        comma(out);
        push_usize_field(
            out,
            "learning_rate_shift",
            usize::from(step.learning_rate_shift),
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

fn push_lexeme_generation_steps_field(
    out: &mut String,
    name: &str,
    steps: &[LexemeGenerationStepTrace],
) {
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
    push_bool_field(out, "corpus_prior", config.decode.corpus_prior);
    comma(out);
    push_usize_field(
        out,
        "corpus_prior_logit_shift",
        usize::from(config.decode.corpus_prior_logit_shift),
    );
    comma(out);
    push_bool_field(out, "strict_adjacency", config.decode.strict_adjacency);
    out.push('}');
}

fn push_lexeme_decode_config_field(out: &mut String, name: &str, config: LexemeGenerationConfig) {
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
    push_bool_field(out, "corpus_prior", config.decode.corpus_prior);
    comma(out);
    push_usize_field(
        out,
        "corpus_prior_logit_shift",
        usize::from(config.decode.corpus_prior_logit_shift),
    );
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

fn push_lexeme_decode_priors_field(
    out: &mut String,
    name: &str,
    trace: Option<LexemeDecodePriorTrace>,
) {
    push_quoted(out, name);
    out.push(':');
    if let Some(trace) = trace {
        out.push('{');
        push_usize_field(out, "token_count", trace.token_count);
        comma(out);
        push_hash_field(out, "token_hash", trace.token_hash);
        comma(out);
        push_usize_field(out, "vocab_size", trace.vocab_size);
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
    push_usize_field(out, "repeat_run", stats.repeat_run);
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

fn push_u16_array_field(out: &mut String, name: &str, values: &[u16]) {
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

    fn encode_u16_test_tokens(tokens: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        bytes
    }

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
    fn lexeme_embedding_training_moves_context_pairs() {
        let tokens =
            encode_u16_test_tokens(&[256, 300, 256, 300, 400, 401, 400, 401, 256, 300, 256, 300]);
        let config = LexemeEmbeddingTrainConfig {
            epochs: 1,
            context_radius: 1,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            vocab_size: 512,
            embedding_dim: 8,
            learning_rate: 1,
            learning_rate_shift: 8,
            concept_frequency_cap: 0,
            concept_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            quality_weight_profile: LexemeQualityWeightProfile::Off,
        };
        let trace = run_lexeme_embedding_training(&tokens, config).expect("lexeme train");

        assert_eq!(trace.token_count, 12);
        assert_eq!(trace.windows, 8);
        assert_eq!(trace.examined_windows, 8);
        assert_eq!(trace.positive_pair_count, trace.negative_pair_count);
        assert_eq!(trace.updates, trace.positive_pair_count);
        assert_eq!(trace.saturation_count, 0);
        assert!(trace.embedding_delta_l1 > 0);
        assert_ne!(trace.initial_embedding_hash, trace.final_embedding_hash);
        assert!(trace.final_positive_dot_i64 > trace.initial_positive_dot_i64);
        assert!(trace.final_negative_dot_i64 < trace.initial_negative_dot_i64);
        assert!(!trace.steps.is_empty());
        assert!(trace.steps[0].positive_dot_after_i64 > trace.steps[0].positive_dot_before_i64);
        if trace.steps[0].negative_dot_before_i64 > LEXEME_NEGATIVE_DOT_MARGIN_I64 {
            assert!(trace.steps[0].negative_dot_after_i64 < trace.steps[0].negative_dot_before_i64);
        }
        let line = trace.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.training_lexeme_embedding_trace.v1\""));
        assert!(line.contains("\"tokenizer\":\"lexeme_ascii_lower_u16_v1\""));
        assert!(line.contains("\"trained_component\":\"lexeme_embedding_i16\""));
    }

    #[test]
    fn lexeme_embedding_model_round_trips() {
        let tokens = encode_u16_test_tokens(&[256, 300, 256, 300, 400, 401, 400, 401]);
        let config = LexemeEmbeddingTrainConfig {
            epochs: 1,
            context_radius: 1,
            stride: 1,
            window_offset: 0,
            max_windows: Some(4),
            vocab_size: 512,
            embedding_dim: 8,
            learning_rate: 1,
            learning_rate_shift: 8,
            concept_frequency_cap: 0,
            concept_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            quality_weight_profile: LexemeQualityWeightProfile::Off,
        };
        let run =
            run_lexeme_embedding_training_with_model(&tokens, config).expect("lexeme model train");
        let bytes = run.model.to_bytes();
        let decoded = LexemeEmbeddingModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.embedding_hash(), run.trace.final_embedding_hash);
        assert_eq!(decoded.vocab_size, config.vocab_size);
        assert_eq!(decoded.embedding_dim, config.embedding_dim);
    }

    #[test]
    fn lexeme_frequency_weights_downweight_repetitive_concepts() {
        let tokens = [256, 256, 256, 256, 300, 301];
        let weights = lexeme_frequency_weights_q15(&tokens, 512, 2, 4096).expect("weights");

        assert!(weights[256] < i16::MAX);
        assert_eq!(weights[300], i16::MAX);
        assert_eq!(weights[301], i16::MAX);
        assert!(weights[256] >= 4096);
    }

    #[test]
    fn lexeme_quality_weights_downweight_document_cruft_without_deleting_it() {
        let mut vocab = vec![String::new(); 260];
        vocab[256] = "class".to_string();
        vocab[257] = "king".to_string();
        vocab[258] = "gutenberg".to_string();
        vocab[259] = "www".to_string();

        let weights =
            lexeme_quality_weights_from_vocab(&vocab, 260, LexemeQualityWeightProfile::CruftAware)
                .expect("weights");

        assert_eq!(weights[256], 4096);
        assert_eq!(weights[257], i16::MAX);
        assert_eq!(weights[258], 4096);
        assert_eq!(weights[259], 4096);
    }

    #[test]
    fn lexeme_embedding_training_traces_concept_frequency_weight() {
        let tokens = encode_u16_test_tokens(&[256, 256, 256, 256, 300, 301, 302, 303]);
        let trace = run_lexeme_embedding_training(
            &tokens,
            LexemeEmbeddingTrainConfig {
                epochs: 1,
                context_radius: 1,
                stride: 1,
                window_offset: 0,
                max_windows: Some(2),
                vocab_size: 512,
                embedding_dim: 8,
                learning_rate: 1,
                learning_rate_shift: 8,
                concept_frequency_cap: 2,
                concept_frequency_min_weight_q15: 4096,
                quality_weight_profile: LexemeQualityWeightProfile::Off,
            },
        )
        .expect("lexeme train");

        assert!(trace.steps[0].positive_frequency_weight_q15 < i16::MAX);
        let line = trace.to_json_line();
        assert!(line.contains("\"concept_frequency_cap\":2"));
        assert!(line.contains("\"positive_frequency_weight_q15\""));
    }

    #[test]
    fn lexeme_embedding_training_traces_quality_weight() {
        let tokens = encode_u16_test_tokens(&[256, 257, 256, 257, 300, 301]);
        let mut quality = vec![i16::MAX; 512];
        quality[256] = 4096;
        let run = run_lexeme_embedding_training_with_model_and_quality(
            &tokens,
            LexemeEmbeddingTrainConfig {
                epochs: 1,
                context_radius: 1,
                stride: 1,
                window_offset: 0,
                max_windows: Some(2),
                vocab_size: 512,
                embedding_dim: 8,
                learning_rate: 1,
                learning_rate_shift: 8,
                concept_frequency_cap: 0,
                concept_frequency_min_weight_q15: 4096,
                quality_weight_profile: LexemeQualityWeightProfile::CruftAware,
            },
            Some(&quality),
        )
        .expect("lexeme train");

        assert_eq!(run.trace.steps[0].positive_quality_weight_q15, 4096);
        assert_eq!(run.trace.steps[0].positive_update_weight_q15, 4096);
        let line = run.trace.to_json_line();
        assert!(line.contains("\"quality_weight_profile\":\"cruft-aware\""));
        assert!(line.contains("\"positive_quality_weight_q15\""));
    }

    #[test]
    fn lexeme_softmax_training_updates_head_with_frozen_embeddings() {
        let tokens =
            encode_u16_test_tokens(&[256, 300, 256, 300, 256, 300, 256, 300, 256, 300, 256, 300]);
        let embeddings = initial_lexeme_embeddings(512, 8).expect("embeddings");
        let embedding_model = LexemeEmbeddingModel::new(512, 8, embeddings).expect("model");
        let initial_embedding_hash = embedding_model.embedding_hash();
        let trace = run_lexeme_softmax_training(
            &tokens,
            embedding_model,
            LexemeSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 2,
                stride: 1,
                window_offset: 0,
                max_windows: Some(10),
                learning_rate: 1,
                learning_rate_shift: 20,
                lr_shift_decay_windows: 0,
                lr_shift_decay_step: 1,
                max_learning_rate_shift: 20,
                max_weight_delta: 1,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                quality_weight_profile: LexemeQualityWeightProfile::Off,
            },
        )
        .expect("lexeme softmax train");

        assert_eq!(trace.token_count, 12);
        assert_eq!(trace.windows, 10);
        assert_eq!(trace.initial_embedding_hash, initial_embedding_hash);
        assert_eq!(trace.final_embedding_hash, initial_embedding_hash);
        assert_ne!(trace.initial_weight_hash, trace.final_weight_hash);
        assert!(trace.weight_delta_l1 > 0);
        assert!(!trace.steps.is_empty());
        assert!(
            trace.steps[0].target_probability_after_q15
                > trace.steps[0].target_probability_before_q15
        );
        assert_eq!(trace.steps[0].learning_rate_shift, 20);
        let line = trace.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.training_lexeme_softmax_trace.v1\""));
        assert!(line.contains("\"model\":{\"id\":\"lexeme_softmax_embedding_head_v1\""));
    }

    #[test]
    fn lexeme_softmax_training_traces_target_frequency_weight() {
        let tokens = encode_u16_test_tokens(&[256, 256, 256, 256, 300, 301, 302, 303]);
        let embeddings = initial_lexeme_embeddings(512, 8).expect("embeddings");
        let embedding_model = LexemeEmbeddingModel::new(512, 8, embeddings).expect("model");
        let trace = run_lexeme_softmax_training(
            &tokens,
            embedding_model,
            LexemeSoftmaxTrainConfig {
                epochs: 1,
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: Some(3),
                learning_rate: 1,
                learning_rate_shift: 20,
                lr_shift_decay_windows: 0,
                lr_shift_decay_step: 1,
                max_learning_rate_shift: 20,
                max_weight_delta: 1,
                target_frequency_cap: 2,
                target_frequency_min_weight_q15: 4096,
                quality_weight_profile: LexemeQualityWeightProfile::Off,
            },
        )
        .expect("lexeme softmax train");

        assert!(trace.steps[0].target_frequency_weight_q15 < i16::MAX);
        let line = trace.to_json_line();
        assert!(line.contains("\"target_frequency_cap\":2"));
        assert!(line.contains("\"target_frequency_weight_q15\""));
    }

    #[test]
    fn lexeme_softmax_training_traces_quality_weight() {
        let tokens = encode_u16_test_tokens(&[256, 256, 256, 256, 300, 301, 302, 303]);
        let embeddings = initial_lexeme_embeddings(512, 8).expect("embeddings");
        let embedding_model = LexemeEmbeddingModel::new(512, 8, embeddings).expect("model");
        let mut quality = vec![i16::MAX; 512];
        quality[256] = 4096;
        let run = run_lexeme_softmax_training_with_model_and_quality(
            &tokens,
            embedding_model,
            LexemeSoftmaxTrainConfig {
                epochs: 1,
                seq_len: 1,
                stride: 1,
                window_offset: 0,
                max_windows: Some(3),
                learning_rate: 1,
                learning_rate_shift: 20,
                lr_shift_decay_windows: 0,
                lr_shift_decay_step: 1,
                max_learning_rate_shift: 20,
                max_weight_delta: 1,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: 4096,
                quality_weight_profile: LexemeQualityWeightProfile::CruftAware,
            },
            Some(&quality),
        )
        .expect("lexeme softmax train");

        assert_eq!(run.trace.steps[0].target_quality_weight_q15, 4096);
        assert_eq!(run.trace.steps[0].target_update_weight_q15, 4096);
        let line = run.trace.to_json_line();
        assert!(line.contains("\"quality_weight_profile\":\"cruft-aware\""));
        assert!(line.contains("\"target_quality_weight_q15\""));
    }

    #[test]
    fn lexeme_softmax_model_round_trips_and_generates() {
        let tokens =
            encode_u16_test_tokens(&[256, 300, 256, 300, 256, 300, 256, 300, 256, 300, 256, 300]);
        let embeddings = initial_lexeme_embeddings(512, 8).expect("embeddings");
        let embedding_model = LexemeEmbeddingModel::new(512, 8, embeddings).expect("model");
        let run = run_lexeme_softmax_training_with_model(
            &tokens,
            embedding_model,
            LexemeSoftmaxTrainConfig {
                epochs: 2,
                seq_len: 2,
                stride: 1,
                window_offset: 0,
                max_windows: Some(10),
                learning_rate: 1,
                learning_rate_shift: 20,
                lr_shift_decay_windows: 0,
                lr_shift_decay_step: 1,
                max_learning_rate_shift: 20,
                max_weight_delta: 1,
                target_frequency_cap: 0,
                target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
                quality_weight_profile: LexemeQualityWeightProfile::Off,
            },
        )
        .expect("lexeme softmax train");
        let bytes = run.model.to_bytes();
        let decoded = LexemeSoftmaxModel::from_bytes(&bytes).expect("model");
        let generation = generate_lexeme_softmax(
            &decoded,
            &[256],
            LexemeGenerationConfig::deterministic_sample(4, 7, 8),
        )
        .expect("generate");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.model_hash(), run.model.model_hash());
        assert_eq!(generation.generated_tokens.len(), 4);
        assert_eq!(generation.model_hash, decoded.model_hash());
        assert_eq!(generation.embedding_hash, decoded.embedding_hash());
        assert_eq!(generation.output_weight_hash, decoded.output_weight_hash());
        let line = generation.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.lexeme_generation_trace.v1\""));
        assert!(line.contains("\"tokenizer\":\"lexeme_ascii_lower_u16_v1\""));
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
                window_offset: 0,
                max_windows: Some(12),
                tokenizer_id: ByteTokenizerId::Identity,
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
            window_offset: 0,
            max_windows: Some(6),
            tokenizer_id: ByteTokenizerId::Identity,
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
    fn byte_softmax_window_offset_selects_later_slice() {
        let tokens = b"0123456789";
        let base = ByteSoftmaxTrainConfig {
            epochs: 1,
            seq_len: 2,
            stride: 2,
            window_offset: 0,
            max_windows: Some(2),
            tokenizer_id: ByteTokenizerId::Identity,
            learning_rate: 1,
            learning_rate_shift: 25,
        };
        let shifted = ByteSoftmaxTrainConfig {
            window_offset: 3,
            ..base
        };

        let base_trace = run_byte_softmax_training(tokens, base).expect("base");
        let shifted_trace = run_byte_softmax_training(tokens, shifted).expect("shifted");

        assert_eq!(base_trace.steps[0].window_start, 0);
        assert_eq!(shifted_trace.steps[0].window_start, 3);
        assert_eq!(shifted_trace.steps[0].last_token, b'4');
        assert_eq!(shifted_trace.steps[0].target_token, b'5');
        assert_ne!(base_trace.window_hash, shifted_trace.window_hash);
        assert!(shifted_trace.to_json_line().contains("\"window_offset\":3"));
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
                window_offset: 0,
                max_windows: Some(12),
                tokenizer_id: ByteTokenizerId::Identity,
                learning_rate: 1,
                learning_rate_shift: 25,
            },
        )
        .expect("train");
        let bytes = run.model.to_bytes();
        let decoded = ByteSoftmaxModel::from_bytes(&bytes).expect("model");

        assert_eq!(decoded, run.model);
        assert_eq!(decoded.weight_hash(), run.trace.final_weight_hash);

        let generation = generate_byte_softmax(&decoded, b"a", ByteGenerationConfig::greedy(6))
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

        let sequence =
            mini_transformer_embedding_sequence_q15(&embeddings, &position_embeddings, b"aa")
                .expect("sequence");
        let first = &sequence[..MINI_TRANSFORMER_D_MODEL];
        let second = &sequence[MINI_TRANSFORMER_D_MODEL..2 * MINI_TRANSFORMER_D_MODEL];

        assert_ne!(first, second);
        assert!(sequence.iter().all(|&value| (-768..=1280).contains(&value)));
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
                window_offset: 0,
                max_windows: Some(12),
                tokenizer_id: ByteTokenizerId::Identity,
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
            window_offset: 0,
            max_windows: Some(5),
            tokenizer_id: ByteTokenizerId::Identity,
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
                window_offset: 0,
                max_windows: Some(12),
                tokenizer_id: ByteTokenizerId::Identity,
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

        let generation =
            generate_byte_embed_softmax(&decoded, b"ab", ByteGenerationConfig::greedy(6))
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
                window_offset: 0,
                max_windows: Some(64),
                batch_windows: 1,
                tokenizer_id: ByteTokenizerId::Identity,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_qk_learning_rate_shift: 18,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
            },
        )
        .expect("mini train");

        assert_eq!(trace.token_count, tokens.len());
        assert!(trace.windows > 0);
        assert!(trace.updates > 0);
        assert!(trace.initial_total_error > trace.final_total_error);
        assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
        assert_ne!(trace.initial_model_hash, trace.final_model_hash);
        assert_ne!(trace.initial_embedding_hash, trace.final_embedding_hash);
        assert_ne!(trace.initial_output_head_hash, trace.final_output_head_hash);
        assert_ne!(trace.initial_mlp_hash, trace.final_mlp_hash);
        assert_eq!(trace.output_head_saturation_count, 0);
        assert!(trace.output_head_delta_l1 > 0);
        assert!(trace.mlp_delta_l1 > 0);
        assert!(trace.embedding_delta_l1 > 0);
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
                .any(|step| step.embedding_hash_before != step.embedding_hash_after)
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
            tokenizer_id: ByteTokenizerId::Identity,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_qk_learning_rate_shift: 18,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
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
    fn mini_transformer_batch_windows_are_traced() {
        let tokens =
            b"To be or not to be, that is the question. To be or not to be, that is the question. ";
        let trace = run_mini_transformer_mlp_training(
            tokens,
            MiniTransformerMlpTrainConfig {
                epochs: 1,
                seq_len: 4,
                stride: 1,
                window_offset: 0,
                max_windows: Some(8),
                batch_windows: 4,
                tokenizer_id: ByteTokenizerId::Identity,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_qk_learning_rate_shift: 18,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
            },
        )
        .expect("mini batch train");

        assert_eq!(trace.examined_windows, 8);
        assert_eq!(trace.accepted_batch_count + trace.rejected_batch_count, 2);
        assert_eq!(trace.mlp_accumulator_window_count, trace.updates);
        assert_eq!(trace.attention_accumulator_window_count, trace.updates);
        assert_eq!(trace.embedding_accumulator_window_count, trace.updates);
        let line = trace.to_json_line();
        assert!(line.contains("\"batch_windows\":4,\"batch_average_shift\":2"));
        assert!(line.contains("\"mlp_accumulator_window_count\""));
        assert!(line.contains("\"attention_accumulator_window_count\""));
        assert!(line.contains("\"embedding_accumulator_window_count\""));
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

    #[test]
    fn attention_weight_gradient_i64_averages_then_updates_i8() {
        let mut embedding_output = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        embedding_output[0] = 4096;
        embedding_output[1] = 8192;
        let mut attention_context = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
        attention_context[0] = 4096;
        attention_context[1] = -4096;
        let cache = MiniTransformerMlpForwardCache {
            attention_norm: embedding_output.clone(),
            embedding_output,
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
            logits_q8: [0_i32; BYTE_VOCAB],
            probabilities_q15: [0_i16; BYTE_VOCAB],
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
                tokenizer_id: ByteTokenizerId::Identity,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 12,
                attention_learning_rate_shift: 22,
                attention_qk_learning_rate_shift: 22,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
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
    fn embedding_gradient_i64_averages_then_updates_i16() {
        let context = [1_u8, 2_u8];
        let mut grad_embedding_output = vec![0_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
        grad_embedding_output[..4].copy_from_slice(&[4096, -4096, 0, 8192]);
        grad_embedding_output[MINI_TRANSFORMER_D_MODEL..MINI_TRANSFORMER_D_MODEL + 4]
            .copy_from_slice(&[-4096, 0, 4096, 0]);
        let mut gradient =
            MiniTransformerEmbeddingGradientI64::new(context.len()).expect("gradient");

        accumulate_mini_transformer_embedding_gradient_i64(
            &context,
            &grad_embedding_output,
            &mut gradient,
        )
        .expect("first sample");
        accumulate_mini_transformer_embedding_gradient_i64(
            &context,
            &grad_embedding_output,
            &mut gradient,
        )
        .expect("second sample");

        let mut embeddings = vec![10_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let mut position_embeddings = vec![10_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
        let stats = apply_mini_transformer_embedding_gradient_i64_to_i16(
            &mut gradient,
            &mut embeddings,
            &mut position_embeddings,
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
    fn attention_vo_oracle_does_not_increase_configured_loss() {
        let tokens = b"To be or not to be, that is the question. To be or not to be. ";
        let seq_len = 4;
        let starts = byte_window_starts(tokens.len(), seq_len, 1, 0, Some(4));
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(seq_len);
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
                tokenizer_id: ByteTokenizerId::Identity,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 16,
                embedding_learning_rate_shift: 14,
                attention_learning_rate_shift: 24,
                attention_qk_learning_rate_shift: 18,
                attention_vo_error_feedback: false,
                attention_vo_oracle: false,
                reject_loss_regression: false,
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
        assert_eq!(generation.context_seq_len, decoded.context_seq_len);
        assert_eq!(generation.model_hash, decoded.model_hash());
        assert_eq!(generation.attention_hash, decoded.attention_hash());
        assert_eq!(generation.mlp_hash, decoded.mlp_hash());
        assert_eq!(generation.output_head_hash, decoded.output_head_hash());
        let line = generation.to_json_line();
        assert!(line.contains("\"schema\":\"nsrl.mini_transformer_generation_trace.v1\""));
        assert!(line.contains("\"model\":\"mini_transformer_byte_qkvo_mlp_v1\""));
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
}
