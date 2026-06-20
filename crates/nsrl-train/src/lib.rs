use nsrl_core::{
    FixedScale, LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace,
    LinearBackwardWeightUpdateI8Params, LinearBackwardWeightUpdateWorkspace, LinearI16I8Params,
    LinearWeightUpdateStats, MAX_RIGHT_SHIFT, base2_softmax_i32_q15,
    linear_backward_input_i16_i8_i16_per_channel_checked, linear_backward_weight_update_i8_checked,
    linear_i16_i8_i16_per_channel_checked, round_shift_rhu_i64, saturate_i8,
};

pub const SCHEMA: &str = "nsrl.training_smoke_trace.v1";
pub const SOFTMAX_SCHEMA: &str = "nsrl.training_softmax_trace.v1";
pub const LINEAR_BACKWARD_SCHEMA: &str = "nsrl.training_linear_backward_trace.v1";
pub const AUTHORITY: &str = "deterministic_training_replay";
pub const TASK: &str = "tiny_next_char_output_head";
pub const SOFTMAX_TASK: &str = "tiny_next_char_output_head_base2_softmax";
pub const LINEAR_BACKWARD_TASK: &str = "tiny_linear_layer_backward";
pub const VOCAB: usize = 4;
pub const D_MODEL: usize = 8;
pub const LINEAR_BACKWARD_INPUT_DIM: usize = 4;
pub const LINEAR_BACKWARD_OUTPUT_DIM: usize = 3;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_EPOCHS: usize = 8;
const DEFAULT_LEARNING_RATE: i8 = 4;
const DEFAULT_SOFTMAX_LEARNING_RATE: i32 = 1;
const DEFAULT_SOFTMAX_LEARNING_RATE_SHIFT: u8 = 22;
const DEFAULT_LINEAR_BACKWARD_LEARNING_RATE: i32 = 1;
const DEFAULT_LINEAR_BACKWARD_LEARNING_RATE_SHIFT: u8 = 22;
const LOGIT_SCALES: [FixedScale; VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; VOCAB];
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
pub enum TrainError {
    InvalidConfig,
    CoreRejected(&'static str),
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid training config"),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoftmaxRow {
    logits_q8: [i32; VOCAB],
    probabilities_q15: [i16; VOCAB],
    softmax_sum_q15: u64,
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

fn hash_i16_slice(values: &[i16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i16_slice(values);
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

fn push_i8_field(out: &mut String, name: &str, value: i8) {
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
}
