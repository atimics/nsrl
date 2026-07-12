#![no_std]
#![deny(unsafe_code)]

#[cfg(test)]
extern crate std;

use core::cmp::Reverse;

use nsrl_core::{
    FixedScale, GatedMlpBackwardScales, GatedMlpBackwardWorkspace, GatedMlpI16Params,
    GatedMlpWeightUpdateParams, GatedMlpWeightUpdateWorkspace, GatedMlpWorkspace,
    LinearAttentionWorkspace, LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace,
    LinearBackwardWeightUpdateI8Params, LinearBackwardWeightUpdateWorkspace, LinearI16I8Params,
    LinearWeightUpdateStats, MAX_RIGHT_SHIFT, Q15_SHIFT, SelfAttentionI16Params,
    base2_softmax_i32_q15, dot_i8_i16_i32_checked, gated_mlp_backward_input_i16_q15_checked,
    gated_mlp_backward_weight_update_i8_checked, gated_mlp_i16_q15_checked, integer_rsqrt_q30,
    linear_attention_i16_q15_checked, linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_backward_weight_update_i8_checked, requantize_i32_to_i16, round_shift_rhu_i64,
    saturate_i8, saturate_i16,
};

pub const BYTE_VOCAB: usize = 256;
// These MUST match the host `MiniTransformerMlpModel` dimensions in the
// `nsrl-train` crate (`MINI_TRANSFORMER_D_MODEL`/`_HEADS`/`_HIDDEN_DIM`). The
// host model is the reference "std" implementation; this no_std core is
// validated against it bit-for-bit by the parity tests in `nsrl-train`
// (`mini_transformer_train_core_linear_nope_step_matches_std_single_window`).
// `mini_transformer_*_train_step` rejects mismatched models with
// `InvalidShape`, which silently zeroes out training, so the two definitions
// must move together. d_model must stay divisible by heads, with a per-head
// dim that is a power of four (128 / 2 = 64 for the default small profile;
// 128 / 8 = 16 for the optional many-head small profile). FixedScale arrays
// below are uniform, so they resize automatically with these constants.
//
// A past scale-up was applied here only and broke that contract; scaling up
// for real means widening this shared source of truth and re-baking
// byte-stable trace fixtures as needed.
pub const MINI_TRANSFORMER_D_MODEL: usize = 128;
#[cfg(not(feature = "mini-heads-8"))]
pub const MINI_TRANSFORMER_HEADS: usize = 2;
#[cfg(feature = "mini-heads-8")]
pub const MINI_TRANSFORMER_HEADS: usize = 8;
pub const MINI_TRANSFORMER_HIDDEN_DIM: usize = 256;
#[cfg(not(feature = "mini-heads-8"))]
pub const MINI_TRANSFORMER_ARCHITECTURE_PROFILE: &str = "small-h2-d128-ff256";
#[cfg(feature = "mini-heads-8")]
pub const MINI_TRANSFORMER_ARCHITECTURE_PROFILE: &str = "small-h8-d128-ff256";
pub const MINI_TRANSFORMER_EMBEDDING_GRAD_FANIN_SHIFT: u8 = 1;

pub const MINI_TRANSFORMER_D_MODEL_SCALES: [FixedScale; MINI_TRANSFORMER_D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    MINI_TRANSFORMER_D_MODEL];
pub const MINI_TRANSFORMER_HIDDEN_SCALES: [FixedScale; MINI_TRANSFORMER_HIDDEN_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 0,
};
    MINI_TRANSFORMER_HIDDEN_DIM];
pub const MINI_TRANSFORMER_OUTPUT_SCALES: [FixedScale; BYTE_VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; BYTE_VOCAB];
pub const MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES: [FixedScale; MINI_TRANSFORMER_D_MODEL] =
    [FixedScale {
        multiplier: 1,
        right_shift: 0,
    }; MINI_TRANSFORMER_D_MODEL];

/// Output (logit) requantization scale shared by forward and backward paths.
///
/// The no-std step API intentionally keeps this fixed. A future calibration
/// controller must update the forward and backward scale tables together rather
/// than overriding only the forward projection.
pub const MINI_TRANSFORMER_DEFAULT_OUTPUT_SCALE: FixedScale = MINI_TRANSFORMER_OUTPUT_SCALES[0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainCoreError {
    InvalidConfig,
    InvalidShape,
    CoreRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerStepConfig {
    pub seq_len: usize,
    pub learning_rate: i32,
    pub output_learning_rate_shift: u8,
    pub mlp_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
    pub attention_learning_rate_shift: u8,
    pub attention_q_learning_rate_shift: u8,
    pub attention_qk_learning_rate_shift: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxUpdateStats {
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedMlpStepStats {
    pub down: LinearWeightUpdateStats,
    pub up: LinearWeightUpdateStats,
    pub gate: LinearWeightUpdateStats,
    pub backward_input_saturation_count: usize,
}

impl GatedMlpStepStats {
    pub fn gradient_saturation_count(self) -> usize {
        self.backward_input_saturation_count
            .saturating_add(self.down.gradient_saturation_count)
            .saturating_add(self.up.gradient_saturation_count)
            .saturating_add(self.gate.gradient_saturation_count)
    }

    pub fn zero_delta_count(self) -> usize {
        self.down
            .zero_delta_count
            .saturating_add(self.up.zero_delta_count)
            .saturating_add(self.gate.zero_delta_count)
    }

    pub fn weight_delta_l1(self) -> u64 {
        self.down
            .weight_delta_l1
            .saturating_add(self.up.weight_delta_l1)
            .saturating_add(self.gate.weight_delta_l1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionStepStats {
    pub q: LinearWeightUpdateStats,
    pub k: LinearWeightUpdateStats,
    pub v: LinearWeightUpdateStats,
    pub o: LinearWeightUpdateStats,
    pub backward_input_saturation_count: usize,
}

impl AttentionStepStats {
    pub fn gradient_saturation_count(self) -> usize {
        self.backward_input_saturation_count
            .saturating_add(self.q.gradient_saturation_count)
            .saturating_add(self.k.gradient_saturation_count)
            .saturating_add(self.v.gradient_saturation_count)
            .saturating_add(self.o.gradient_saturation_count)
    }

    pub fn zero_delta_count(self) -> usize {
        self.q
            .zero_delta_count
            .saturating_add(self.k.zero_delta_count)
            .saturating_add(self.v.zero_delta_count)
            .saturating_add(self.o.zero_delta_count)
    }

    pub fn weight_delta_l1(self) -> u64 {
        self.q
            .weight_delta_l1
            .saturating_add(self.k.weight_delta_l1)
            .saturating_add(self.v.weight_delta_l1)
            .saturating_add(self.o.weight_delta_l1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerStepStats {
    pub predicted_before: u8,
    pub predicted_after: u8,
    pub residual_saturation_count: usize,
    pub output_head: LinearWeightUpdateStats,
    pub mlp: GatedMlpStepStats,
    pub attention: AttentionStepStats,
    pub embedding: SoftmaxUpdateStats,
}

pub struct MiniTransformerModelSlicesMut<'a> {
    pub embeddings: &'a mut [i16],
    pub q_weights: &'a mut [i8],
    pub k_weights: &'a mut [i8],
    pub v_weights: &'a mut [i8],
    pub o_weights: &'a mut [i8],
    pub up_weights: &'a mut [i8],
    pub gate_weights: &'a mut [i8],
    pub down_weights: &'a mut [i8],
    pub output_weights: &'a mut [i8],
}

pub struct MiniTransformerStepWorkspace<'a> {
    pub embedding_output: &'a mut [i16],
    pub attention_norm: &'a mut [i16],
    pub attention_q: &'a mut [i16],
    pub attention_k: &'a mut [i16],
    pub attention_v: &'a mut [i16],
    pub attention_context: &'a mut [i16],
    pub attention_output: &'a mut [i16],
    pub attention_residual: &'a mut [i16],
    pub attention_state_kv: &'a mut [i64],
    pub attention_key_sums: &'a mut [i64],
    pub mlp_norm: &'a mut [i16],
    pub mlp_up: &'a mut [i16],
    pub mlp_gate: &'a mut [i16],
    pub mlp_gated: &'a mut [i16],
    pub mlp_output: &'a mut [i16],
    pub block_output: &'a mut [i16],
    pub logits_q8: &'a mut [i32],
    pub probabilities_q15: &'a mut [i16],
    pub grad_output_q15: &'a mut [i16],
    pub output_scaled_grad: &'a mut [i32],
    pub grad_last_features: &'a mut [i16],
    pub grad_mlp_output: &'a mut [i16],
    pub grad_mlp_input: &'a mut [i16],
    pub mlp_scaled_grad: &'a mut [i32],
    pub mlp_input_grad_gated: &'a mut [i16],
    pub mlp_input_grad_up: &'a mut [i16],
    pub mlp_input_grad_gate: &'a mut [i16],
    pub mlp_input_grad_up_input: &'a mut [i16],
    pub mlp_input_grad_gate_input: &'a mut [i16],
    pub mlp_update_grad_gated: &'a mut [i16],
    pub mlp_update_grad_up: &'a mut [i16],
    pub mlp_update_grad_gate: &'a mut [i16],
    pub grad_attention_output: &'a mut [i16],
    pub grad_attention_context: &'a mut [i16],
    pub attention_scaled_grad: &'a mut [i32],
    pub linear_prefix_states: &'a mut [i64],
    pub linear_denominators: &'a mut [i64],
    pub linear_grad_state_q15: &'a mut [i64],
    pub linear_grad_q_acc: &'a mut [i64],
    pub linear_grad_k_acc: &'a mut [i64],
    pub linear_grad_v_acc: &'a mut [i64],
    pub grad_attention_q: &'a mut [i16],
    pub grad_attention_k: &'a mut [i16],
    pub grad_attention_v: &'a mut [i16],
    pub grad_attention_norm_input: &'a mut [i16],
    pub grad_embedding_output: &'a mut [i16],
}

pub struct LinearWeightGradientI64Workspace<'a> {
    pub input_dim: usize,
    pub output_dim: usize,
    pub sample_count: usize,
    pub accumulators: &'a mut [i64],
    pub residuals: &'a mut [i64],
}

/// Bounded, deterministic Adam-style optimizer parameters.
///
/// `step_shift` is applied after the Q15-normalized moment ratio, so an
/// effective weight step uses `Q15_SHIFT + step_shift`. The beta shifts encode
/// `beta1 = 1 - 2^-beta1_decay_shift` and
/// `beta2 = 1 - 2^-beta2_decay_shift` without floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerAdamConfig {
    pub learning_rate: i32,
    pub step_shift: u8,
    pub beta1_decay_shift: u8,
    pub beta2_decay_shift: u8,
    pub epsilon: u64,
}

impl Default for IntegerAdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1,
            step_shift: 4,
            beta1_decay_shift: 3,
            beta2_decay_shift: 7,
            epsilon: 1,
        }
    }
}

impl IntegerAdamConfig {
    pub fn is_valid(self) -> bool {
        self.learning_rate > 0
            && self.beta1_decay_shift > 0
            && self.beta2_decay_shift > 0
            && self.beta1_decay_shift <= MAX_RIGHT_SHIFT
            && self.beta2_decay_shift <= MAX_RIGHT_SHIFT
            && self.step_shift <= MAX_RIGHT_SHIFT.saturating_sub(Q15_SHIFT)
            && self.epsilon > 0
    }
}

/// Mutable optimizer state kept separately from inference weights.
pub struct IntegerAdamStateWorkspace<'a> {
    pub step: u64,
    pub first_moments: &'a mut [i64],
    pub second_moments: &'a mut [u64],
    pub update_residuals: &'a mut [i64],
}

impl IntegerAdamStateWorkspace<'_> {
    pub fn is_valid_for(&self, len: usize) -> bool {
        self.first_moments.len() == len
            && self.second_moments.len() == len
            && self.update_residuals.len() == len
    }
}

impl LinearWeightGradientI64Workspace<'_> {
    pub fn is_valid(&self) -> bool {
        self.input_dim != 0
            && self.output_dim != 0
            && self
                .input_dim
                .checked_mul(self.output_dim)
                .is_some_and(|len| self.accumulators.len() == len && self.residuals.len() == len)
    }

    pub fn clear(&mut self) {
        self.sample_count = 0;
        self.accumulators.fill(0);
    }
}

pub fn accumulate_linear_weight_gradient_i64_prescaled(
    input: &[i16],
    scaled_grad_output: &[i32],
    gradient: &mut LinearWeightGradientI64Workspace<'_>,
) -> Result<(), TrainCoreError> {
    if !gradient.is_valid()
        || input.len() != gradient.input_dim
        || scaled_grad_output.len() != gradient.output_dim
    {
        return Err(TrainCoreError::InvalidShape);
    }

    for (out_index, &scaled_grad) in scaled_grad_output.iter().enumerate() {
        if scaled_grad == 0 {
            continue;
        }
        let row_start = out_index
            .checked_mul(gradient.input_dim)
            .ok_or(TrainCoreError::CoreRejected)?;
        for (in_index, &activation) in input.iter().enumerate() {
            if activation == 0 {
                continue;
            }
            let product = i64::from(scaled_grad)
                .checked_mul(i64::from(activation))
                .ok_or(TrainCoreError::CoreRejected)?;
            let index = row_start
                .checked_add(in_index)
                .ok_or(TrainCoreError::CoreRejected)?;
            gradient.accumulators[index] = gradient.accumulators[index]
                .checked_add(product)
                .ok_or(TrainCoreError::CoreRejected)?;
        }
    }

    gradient.sample_count = gradient
        .sample_count
        .checked_add(1)
        .ok_or(TrainCoreError::CoreRejected)?;
    Ok(())
}

pub fn apply_linear_weight_gradient_i64_to_i8(
    gradient: &mut LinearWeightGradientI64Workspace<'_>,
    weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<LinearWeightUpdateStats, TrainCoreError> {
    if !gradient.is_valid()
        || weights.len() != gradient.accumulators.len()
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainCoreError::InvalidConfig);
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
        let product = averaged
            .checked_mul(i64::from(learning_rate))
            .ok_or(TrainCoreError::CoreRejected)?;
        let product = if carry_residual {
            product
                .checked_add(*residual)
                .ok_or(TrainCoreError::CoreRejected)?
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
            stats.zero_delta_count = stats
                .zero_delta_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
        }

        let previous = *weight;
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainCoreError::CoreRejected)?;
        let clamped = saturate_i8(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
            *residual = 0;
        } else {
            *residual = next_residual;
        }
        let applied_delta = i64::from(clamped) - i64::from(previous);
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .checked_add(applied_delta.unsigned_abs())
            .ok_or(TrainCoreError::CoreRejected)?;
        *weight = clamped;
    }

    gradient.clear();
    Ok(stats)
}

/// Applies averaged i64 gradients to i8 weights with integer Adam-style
/// momentum, variance normalization, and sub-weight error feedback.
pub fn apply_integer_adam_accumulators_i64_to_i8(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i8],
    config: IntegerAdamConfig,
    state: &mut IntegerAdamStateWorkspace<'_>,
) -> Result<LinearWeightUpdateStats, TrainCoreError> {
    if accumulators.len() != weights.len() || !state.is_valid_for(weights.len()) {
        return Err(TrainCoreError::InvalidShape);
    }
    validate_integer_adam_config(config)?;
    if sample_count == 0 {
        return Ok(empty_linear_weight_update_stats());
    }

    let mut stats = empty_linear_weight_update_stats();
    for index in 0..weights.len() {
        let gradient = round_div_i64(accumulators[index], sample_count)?;
        let (delta, saturated_gradient) = integer_adam_delta(
            gradient,
            config,
            &mut state.first_moments[index],
            &mut state.second_moments[index],
            &mut state.update_residuals[index],
        )?;
        if saturated_gradient {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
        }
        if delta == 0 {
            stats.zero_delta_count = stats
                .zero_delta_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
        }
        let previous = weights[index];
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainCoreError::CoreRejected)?;
        let clamped = saturate_i8(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
            state.update_residuals[index] = 0;
        }
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .checked_add((i64::from(clamped) - i64::from(previous)).unsigned_abs())
            .ok_or(TrainCoreError::CoreRejected)?;
        weights[index] = clamped;
    }
    state.step = state
        .step
        .checked_add(1)
        .ok_or(TrainCoreError::CoreRejected)?;
    Ok(stats)
}

/// i16 counterpart used by token and position embeddings and future RMSNorm
/// scale vectors.
pub fn apply_integer_adam_accumulators_i64_to_i16(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i16],
    config: IntegerAdamConfig,
    state: &mut IntegerAdamStateWorkspace<'_>,
) -> Result<LinearWeightUpdateStats, TrainCoreError> {
    if accumulators.len() != weights.len() || !state.is_valid_for(weights.len()) {
        return Err(TrainCoreError::InvalidShape);
    }
    validate_integer_adam_config(config)?;
    if sample_count == 0 {
        return Ok(empty_linear_weight_update_stats());
    }

    let mut stats = empty_linear_weight_update_stats();
    for index in 0..weights.len() {
        let gradient = round_div_i64(accumulators[index], sample_count)?;
        let (delta, saturated_gradient) = integer_adam_delta(
            gradient,
            config,
            &mut state.first_moments[index],
            &mut state.second_moments[index],
            &mut state.update_residuals[index],
        )?;
        if saturated_gradient {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
        }
        if delta == 0 {
            stats.zero_delta_count = stats
                .zero_delta_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
        }
        let previous = weights[index];
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainCoreError::CoreRejected)?;
        let clamped = saturate_i16(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats
                .gradient_saturation_count
                .checked_add(1)
                .ok_or(TrainCoreError::CoreRejected)?;
            state.update_residuals[index] = 0;
        }
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .checked_add((i64::from(clamped) - i64::from(previous)).unsigned_abs())
            .ok_or(TrainCoreError::CoreRejected)?;
        weights[index] = clamped;
    }
    state.step = state
        .step
        .checked_add(1)
        .ok_or(TrainCoreError::CoreRejected)?;
    Ok(stats)
}

fn validate_integer_adam_config(config: IntegerAdamConfig) -> Result<(), TrainCoreError> {
    if !config.is_valid() {
        return Err(TrainCoreError::InvalidConfig);
    }
    Ok(())
}

fn integer_adam_delta(
    gradient: i64,
    config: IntegerAdamConfig,
    first_moment: &mut i64,
    second_moment: &mut u64,
    update_residual: &mut i64,
) -> Result<(i64, bool), TrainCoreError> {
    // Squaring an arbitrary i64 can overflow u64. This explicit cap keeps the
    // moment domain stable and makes saturation observable in update stats.
    const MAX_MOMENT_GRADIENT: i64 = i32::MAX as i64;
    let bounded_gradient = gradient.clamp(-MAX_MOMENT_GRADIENT, MAX_MOMENT_GRADIENT);
    let saturated_gradient = bounded_gradient != gradient;

    let beta1_denominator = 1_i128
        .checked_shl(u32::from(config.beta1_decay_shift))
        .ok_or(TrainCoreError::InvalidConfig)?;
    let next_first_numerator = i128::from(*first_moment)
        .checked_mul(beta1_denominator - 1)
        .and_then(|value| value.checked_add(i128::from(bounded_gradient)))
        .ok_or(TrainCoreError::CoreRejected)?;
    *first_moment = round_shift_rhu_i128_to_i64(next_first_numerator, config.beta1_decay_shift)?;

    let gradient_square = i128::from(bounded_gradient)
        .checked_mul(i128::from(bounded_gradient))
        .ok_or(TrainCoreError::CoreRejected)?;
    let beta2_denominator = 1_i128
        .checked_shl(u32::from(config.beta2_decay_shift))
        .ok_or(TrainCoreError::InvalidConfig)?;
    let next_second_numerator = i128::from(*second_moment)
        .checked_mul(beta2_denominator - 1)
        .and_then(|value| value.checked_add(gradient_square))
        .ok_or(TrainCoreError::CoreRejected)?;
    *second_moment = u64::try_from(round_shift_rhu_i128(
        next_second_numerator,
        config.beta2_decay_shift,
    )?)
    .map_err(|_| TrainCoreError::CoreRejected)?;

    let variance_with_epsilon = second_moment
        .checked_add(config.epsilon)
        .ok_or(TrainCoreError::CoreRejected)?;
    let inverse_root_q30 =
        integer_rsqrt_q30(variance_with_epsilon).ok_or(TrainCoreError::CoreRejected)?;
    let normalized_q30 = i128::from(*first_moment)
        .checked_mul(i128::from(inverse_root_q30))
        .ok_or(TrainCoreError::CoreRejected)?;
    let normalized_q15 = round_shift_rhu_i128_to_i64(normalized_q30, Q15_SHIFT)?
        .clamp(-i64::from(i16::MAX), i64::from(i16::MAX));
    let update_numerator = normalized_q15
        .checked_mul(i64::from(config.learning_rate))
        .and_then(|value| value.checked_add(*update_residual))
        .ok_or(TrainCoreError::CoreRejected)?;
    let effective_shift = Q15_SHIFT
        .checked_add(config.step_shift)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let scaled_update = round_shift_rhu_i64(update_numerator, effective_shift);
    *update_residual =
        rounded_shift_residual_i64(update_numerator, scaled_update, effective_shift)?;
    Ok((-scaled_update, saturated_gradient))
}

fn round_shift_rhu_i128(value: i128, right_shift: u8) -> Result<i128, TrainCoreError> {
    if right_shift == 0 {
        return Ok(value);
    }
    let offset = 1_i128
        .checked_shl(u32::from(right_shift - 1))
        .ok_or(TrainCoreError::InvalidConfig)?;
    value
        .checked_add(offset)
        .map(|rounded| rounded >> right_shift)
        .ok_or(TrainCoreError::CoreRejected)
}

fn round_shift_rhu_i128_to_i64(value: i128, right_shift: u8) -> Result<i64, TrainCoreError> {
    i64::try_from(round_shift_rhu_i128(value, right_shift)?)
        .map_err(|_| TrainCoreError::CoreRejected)
}

pub fn mini_transformer_linear_nope_train_step(
    model: &mut MiniTransformerModelSlicesMut<'_>,
    context: &[u8],
    target: u8,
    config: MiniTransformerStepConfig,
    workspace: &mut MiniTransformerStepWorkspace<'_>,
) -> Result<MiniTransformerStepStats, TrainCoreError> {
    validate_config(config)?;
    validate_model_shapes(model, config.seq_len)?;
    validate_workspace_shapes(workspace, config.seq_len)?;

    let forward_before_residual_saturation_count =
        mini_transformer_forward_linear_nope(model, context, config.seq_len, workspace)?;
    let predicted_before = byte_argmax_i32(workspace.logits_q8)?;

    byte_vocab_softmax_gradient_q15(
        target,
        workspace.probabilities_q15,
        workspace.grad_output_q15,
    )?;
    linear_backward_input_i16_i8_i16_per_channel_checked(
        workspace.grad_output_q15,
        LinearBackwardInputI16I8Params {
            weights: model.output_weights,
            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
            grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: BYTE_VOCAB,
        },
        LinearBackwardInputWorkspace {
            scaled_grad_output: workspace.output_scaled_grad,
        },
        workspace.grad_last_features,
    )
    .ok_or(TrainCoreError::CoreRejected)?;

    let last_start = (config.seq_len - 1)
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let last_end = last_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let output_head = linear_backward_weight_update_i8_checked(
        &workspace.block_output[last_start..last_end],
        workspace.grad_output_q15,
        model.output_weights,
        LinearBackwardWeightUpdateI8Params {
            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: BYTE_VOCAB,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.output_learning_rate_shift,
        },
        LinearBackwardWeightUpdateWorkspace {
            scaled_grad_output: workspace.output_scaled_grad,
        },
    )
    .ok_or(TrainCoreError::CoreRejected)?;

    workspace.grad_mlp_output.fill(0);
    workspace.grad_mlp_output[last_start..last_end].copy_from_slice(workspace.grad_last_features);
    let mlp_backward_input_saturation = gated_mlp_backward_input_i16_q15_checked(
        workspace.grad_mlp_output,
        mini_transformer_mlp_params(model, config.seq_len),
        workspace.mlp_up,
        workspace.mlp_gate,
        GatedMlpBackwardScales {
            down_to_hidden: &MINI_TRANSFORMER_HIDDEN_SCALES,
            up_to_input: &MINI_TRANSFORMER_D_MODEL_SCALES,
            gate_to_input: &MINI_TRANSFORMER_D_MODEL_SCALES,
        },
        GatedMlpBackwardWorkspace {
            scaled_grad_output: workspace.mlp_scaled_grad,
            grad_gated: workspace.mlp_input_grad_gated,
            grad_up: workspace.mlp_input_grad_up,
            grad_gate: workspace.mlp_input_grad_gate,
            grad_up_input: workspace.mlp_input_grad_up_input,
            grad_gate_input: workspace.mlp_input_grad_gate_input,
        },
        workspace.grad_mlp_input,
    )
    .ok_or(TrainCoreError::CoreRejected)?;

    let gradient_residual_saturation = add_i16_residual_rows_checked(
        workspace.grad_mlp_output,
        workspace.grad_mlp_input,
        workspace.grad_attention_output,
    )?;

    let mlp_update = gated_mlp_backward_weight_update_i8_checked(
        workspace.mlp_norm,
        workspace.grad_mlp_output,
        workspace.mlp_up,
        workspace.mlp_gate,
        workspace.mlp_gated,
        model.up_weights,
        model.gate_weights,
        model.down_weights,
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
            scaled_grad_output: workspace.mlp_scaled_grad,
            grad_gated: workspace.mlp_update_grad_gated,
            grad_up: workspace.mlp_update_grad_up,
            grad_gate: workspace.mlp_update_grad_gate,
        },
    )
    .ok_or(TrainCoreError::CoreRejected)?;

    let attention = mini_transformer_linear_attention_update_i8_checked(
        model,
        workspace,
        config.seq_len,
        config.learning_rate,
        config.attention_q_learning_rate_shift,
        config.attention_qk_learning_rate_shift,
        config.attention_learning_rate_shift,
    )?;
    let embedding_gradient_saturation = add_i16_residual_rows_checked(
        workspace.grad_attention_output,
        workspace.grad_attention_norm_input,
        workspace.grad_embedding_output,
    )?;
    let embedding = apply_mini_transformer_embedding_update_nope(
        model.embeddings,
        context,
        workspace.grad_embedding_output,
        config.learning_rate,
        config.embedding_learning_rate_shift,
    )?;

    let forward_after_residual_saturation_count =
        mini_transformer_forward_linear_nope(model, context, config.seq_len, workspace)?;
    let predicted_after = byte_argmax_i32(workspace.logits_q8)?;
    let residual_saturation_count = gradient_residual_saturation
        .saturating_add(embedding_gradient_saturation)
        .saturating_add(forward_before_residual_saturation_count)
        .saturating_add(forward_after_residual_saturation_count);

    Ok(MiniTransformerStepStats {
        predicted_before,
        predicted_after,
        residual_saturation_count,
        output_head,
        mlp: GatedMlpStepStats {
            down: mlp_update.down,
            up: mlp_update.up,
            gate: mlp_update.gate,
            backward_input_saturation_count: mlp_backward_input_saturation,
        },
        attention,
        embedding,
    })
}

fn validate_config(config: MiniTransformerStepConfig) -> Result<(), TrainCoreError> {
    if config.seq_len == 0
        || config.learning_rate <= 0
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_q_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainCoreError::InvalidConfig);
    }
    mini_transformer_head_dim().ok_or(TrainCoreError::InvalidConfig)?;
    Ok(())
}

fn validate_model_shapes(
    model: &MiniTransformerModelSlicesMut<'_>,
    seq_len: usize,
) -> Result<(), TrainCoreError> {
    let attention_weights = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL;
    let up_gate_weights = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM;
    if model.embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || model.q_weights.len() != attention_weights
        || model.k_weights.len() != attention_weights
        || model.v_weights.len() != attention_weights
        || model.o_weights.len() != attention_weights
        || model.up_weights.len() != up_gate_weights
        || model.gate_weights.len() != up_gate_weights
        || model.down_weights.len() != MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL
        || model.output_weights.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || seq_len == 0
    {
        return Err(TrainCoreError::InvalidShape);
    }
    Ok(())
}

fn validate_workspace_shapes(
    workspace: &MiniTransformerStepWorkspace<'_>,
    seq_len: usize,
) -> Result<(), TrainCoreError> {
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let head_dim = mini_transformer_head_dim().ok_or(TrainCoreError::InvalidConfig)?;
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let state_len = MINI_TRANSFORMER_HEADS
        .checked_mul(head_state_len)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let key_sum_len = MINI_TRANSFORMER_HEADS
        .checked_mul(head_dim)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let prefix_len = state_len
        .checked_mul(seq_len)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let denom_len = MINI_TRANSFORMER_HEADS
        .checked_mul(seq_len)
        .ok_or(TrainCoreError::InvalidConfig)?;

    if workspace.embedding_output.len() != total
        || workspace.attention_norm.len() != total
        || workspace.attention_q.len() != total
        || workspace.attention_k.len() != total
        || workspace.attention_v.len() != total
        || workspace.attention_context.len() != total
        || workspace.attention_output.len() != total
        || workspace.attention_residual.len() != total
        || workspace.attention_state_kv.len() != state_len
        || workspace.attention_key_sums.len() != key_sum_len
        || workspace.mlp_norm.len() != total
        || workspace.mlp_up.len() != hidden_total
        || workspace.mlp_gate.len() != hidden_total
        || workspace.mlp_gated.len() != hidden_total
        || workspace.mlp_output.len() != total
        || workspace.block_output.len() != total
        || workspace.logits_q8.len() != BYTE_VOCAB
        || workspace.probabilities_q15.len() != BYTE_VOCAB
        || workspace.grad_output_q15.len() != BYTE_VOCAB
        || workspace.output_scaled_grad.len() != BYTE_VOCAB
        || workspace.grad_last_features.len() != MINI_TRANSFORMER_D_MODEL
        || workspace.grad_mlp_output.len() != total
        || workspace.grad_mlp_input.len() != total
        || workspace.mlp_scaled_grad.len()
            < MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM)
        || workspace.mlp_input_grad_gated.len() != hidden_total
        || workspace.mlp_input_grad_up.len() != hidden_total
        || workspace.mlp_input_grad_gate.len() != hidden_total
        || workspace.mlp_input_grad_up_input.len() != total
        || workspace.mlp_input_grad_gate_input.len() != total
        || workspace.mlp_update_grad_gated.len() != hidden_total
        || workspace.mlp_update_grad_up.len() != hidden_total
        || workspace.mlp_update_grad_gate.len() != hidden_total
        || workspace.grad_attention_output.len() != total
        || workspace.grad_attention_context.len() != total
        || workspace.attention_scaled_grad.len() < MINI_TRANSFORMER_D_MODEL
        || workspace.linear_prefix_states.len() != prefix_len
        || workspace.linear_denominators.len() != denom_len
        || workspace.linear_grad_state_q15.len() != head_state_len
        || workspace.linear_grad_q_acc.len() != total
        || workspace.linear_grad_k_acc.len() != total
        || workspace.linear_grad_v_acc.len() != total
        || workspace.grad_attention_q.len() != total
        || workspace.grad_attention_k.len() != total
        || workspace.grad_attention_v.len() != total
        || workspace.grad_attention_norm_input.len() != total
        || workspace.grad_embedding_output.len() != total
    {
        return Err(TrainCoreError::InvalidShape);
    }
    Ok(())
}

fn mini_transformer_forward_linear_nope(
    model: &MiniTransformerModelSlicesMut<'_>,
    context: &[u8],
    seq_len: usize,
    workspace: &mut MiniTransformerStepWorkspace<'_>,
) -> Result<usize, TrainCoreError> {
    if context.len() != seq_len {
        return Err(TrainCoreError::InvalidConfig);
    }
    mini_transformer_embedding_sequence_nope_q15(
        model.embeddings,
        context,
        workspace.embedding_output,
    )?;
    workspace
        .attention_norm
        .copy_from_slice(workspace.embedding_output);

    let attention_params = mini_transformer_attention_params(model, seq_len);
    workspace.attention_state_kv.fill(0);
    workspace.attention_key_sums.fill(0);
    linear_attention_i16_q15_checked(
        workspace.attention_norm,
        attention_params,
        LinearAttentionWorkspace {
            q: workspace.attention_q,
            k: workspace.attention_k,
            v: workspace.attention_v,
            context: workspace.attention_context,
            state_kv: workspace.attention_state_kv,
            key_sums: workspace.attention_key_sums,
        },
        workspace.attention_output,
    )
    .ok_or(TrainCoreError::CoreRejected)?;

    let mut residual_saturation_count = add_i16_residual_rows_checked(
        workspace.embedding_output,
        workspace.attention_output,
        workspace.attention_residual,
    )?;
    workspace
        .mlp_norm
        .copy_from_slice(workspace.attention_residual);

    gated_mlp_i16_q15_checked(
        workspace.mlp_norm,
        mini_transformer_mlp_params(model, seq_len),
        GatedMlpWorkspace {
            up: workspace.mlp_up,
            gate: workspace.mlp_gate,
            gated: workspace.mlp_gated,
        },
        workspace.mlp_output,
    )
    .ok_or(TrainCoreError::CoreRejected)?;
    residual_saturation_count =
        residual_saturation_count.saturating_add(add_i16_residual_rows_checked(
            workspace.attention_residual,
            workspace.mlp_output,
            workspace.block_output,
        )?);
    let last_start = (seq_len - 1)
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let last_end = last_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    mini_transformer_output_row_for(
        model.output_weights,
        &workspace.block_output[last_start..last_end],
        workspace.logits_q8,
        workspace.probabilities_q15,
    )?;
    Ok(residual_saturation_count)
}

fn mini_transformer_embedding_sequence_nope_q15(
    embeddings: &[i16],
    context: &[u8],
    output: &mut [i16],
) -> Result<(), TrainCoreError> {
    let total = context
        .len()
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL || output.len() != total {
        return Err(TrainCoreError::InvalidShape);
    }
    for (position, &token) in context.iter().enumerate() {
        let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        let out_start = position * MINI_TRANSFORMER_D_MODEL;
        output[out_start..out_start + MINI_TRANSFORMER_D_MODEL]
            .copy_from_slice(&embeddings[row_start..row_start + MINI_TRANSFORMER_D_MODEL]);
    }
    Ok(())
}

fn mini_transformer_attention_params<'a>(
    model: &'a MiniTransformerModelSlicesMut<'_>,
    seq_len: usize,
) -> SelfAttentionI16Params<'a> {
    SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: model.q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: model.k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: model.v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: model.o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    }
}

fn mini_transformer_mlp_params<'a>(
    model: &'a MiniTransformerModelSlicesMut<'_>,
    seq_len: usize,
) -> GatedMlpI16Params<'a> {
    GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: model.up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: model.gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: model.down_weights,
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

fn mini_transformer_output_row_for(
    output_weights: &[i8],
    features: &[i16],
    logits_q8: &mut [i32],
    probabilities_q15: &mut [i16],
) -> Result<(), TrainCoreError> {
    if output_weights.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || features.len() != MINI_TRANSFORMER_D_MODEL
        || logits_q8.len() != BYTE_VOCAB
        || probabilities_q15.len() != BYTE_VOCAB
    {
        return Err(TrainCoreError::InvalidShape);
    }
    for (class_id, logit) in logits_q8.iter_mut().enumerate() {
        let row_start = class_id * MINI_TRANSFORMER_D_MODEL;
        let acc = dot_i8_i16_i32_checked(
            &output_weights[row_start..row_start + MINI_TRANSFORMER_D_MODEL],
            features,
        )
        .ok_or(TrainCoreError::CoreRejected)?;
        *logit = i32::from(requantize_i32_to_i16(
            acc,
            MINI_TRANSFORMER_DEFAULT_OUTPUT_SCALE,
        ));
    }
    base2_softmax_i32_q15(logits_q8, probabilities_q15)
        .map(|_| ())
        .ok_or(TrainCoreError::CoreRejected)
}

fn byte_vocab_softmax_gradient_q15(
    target: u8,
    probabilities_q15: &[i16],
    grad_output_q15: &mut [i16],
) -> Result<(), TrainCoreError> {
    if probabilities_q15.len() != BYTE_VOCAB || grad_output_q15.len() != BYTE_VOCAB {
        return Err(TrainCoreError::InvalidShape);
    }
    let target = usize::from(target);
    for (class_id, out) in grad_output_q15.iter_mut().enumerate() {
        let mut value = i32::from(probabilities_q15[class_id]);
        if class_id == target {
            value -= i32::from(i16::MAX);
        }
        *out = saturate_i16(i64::from(value));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_linear_attention_update_i8_checked(
    model: &mut MiniTransformerModelSlicesMut<'_>,
    workspace: &mut MiniTransformerStepWorkspace<'_>,
    seq_len: usize,
    learning_rate: i32,
    attention_q_learning_rate_shift: u8,
    attention_qk_learning_rate_shift: u8,
    attention_learning_rate_shift: u8,
) -> Result<AttentionStepStats, TrainCoreError> {
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    workspace.grad_attention_context.fill(0);
    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_output[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: model.o_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: workspace.attention_scaled_grad,
            },
            &mut workspace.grad_attention_context[row_start..row_end],
        )
        .ok_or(TrainCoreError::CoreRejected)?;
    }

    mini_transformer_linear_attention_qkv_gradients_q15(seq_len, workspace)?;

    workspace.grad_attention_norm_input.fill(0);
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
                weights: model.q_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: workspace.attention_scaled_grad,
            },
            &mut grad_q_input,
        )
        .ok_or(TrainCoreError::CoreRejected)?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_k[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: model.k_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: workspace.attention_scaled_grad,
            },
            &mut grad_k_input,
        )
        .ok_or(TrainCoreError::CoreRejected)?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_v[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: model.v_weights,
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: workspace.attention_scaled_grad,
            },
            &mut grad_v_input,
        )
        .ok_or(TrainCoreError::CoreRejected)?;

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

    let mut q = empty_linear_weight_update_stats();
    let mut k = empty_linear_weight_update_stats();
    let mut v = empty_linear_weight_update_stats();
    let mut o = empty_linear_weight_update_stats();
    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        q = add_linear_weight_update_stats(
            q,
            linear_backward_weight_update_i8_checked(
                &workspace.attention_norm[row_start..row_end],
                &workspace.grad_attention_q[row_start..row_end],
                model.q_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate,
                    learning_rate_shift: attention_q_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: workspace.attention_scaled_grad,
                },
            )
            .ok_or(TrainCoreError::CoreRejected)?,
        );
        k = add_linear_weight_update_stats(
            k,
            linear_backward_weight_update_i8_checked(
                &workspace.attention_norm[row_start..row_end],
                &workspace.grad_attention_k[row_start..row_end],
                model.k_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate,
                    learning_rate_shift: attention_qk_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: workspace.attention_scaled_grad,
                },
            )
            .ok_or(TrainCoreError::CoreRejected)?,
        );
        v = add_linear_weight_update_stats(
            v,
            linear_backward_weight_update_i8_checked(
                &workspace.attention_norm[row_start..row_end],
                &workspace.grad_attention_v[row_start..row_end],
                model.v_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate,
                    learning_rate_shift: attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: workspace.attention_scaled_grad,
                },
            )
            .ok_or(TrainCoreError::CoreRejected)?,
        );
        o = add_linear_weight_update_stats(
            o,
            linear_backward_weight_update_i8_checked(
                &workspace.attention_context[row_start..row_end],
                &workspace.grad_attention_output[row_start..row_end],
                model.o_weights,
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate,
                    learning_rate_shift: attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: workspace.attention_scaled_grad,
                },
            )
            .ok_or(TrainCoreError::CoreRejected)?,
        );
    }

    debug_assert_eq!(workspace.grad_attention_norm_input.len(), total);
    Ok(AttentionStepStats {
        q,
        k,
        v,
        o,
        backward_input_saturation_count: input_gradient_saturation_count,
    })
}

fn mini_transformer_linear_attention_qkv_gradients_q15(
    seq_len: usize,
    workspace: &mut MiniTransformerStepWorkspace<'_>,
) -> Result<(), TrainCoreError> {
    let head_dim = mini_transformer_head_dim().ok_or(TrainCoreError::InvalidConfig)?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainCoreError::InvalidConfig)?;
    let state_len = MINI_TRANSFORMER_HEADS
        .checked_mul(head_state_len)
        .ok_or(TrainCoreError::InvalidConfig)?;
    workspace.linear_grad_q_acc[..total].fill(0);
    workspace.linear_grad_k_acc[..total].fill(0);
    workspace.linear_grad_v_acc[..total].fill(0);

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainCoreError::InvalidConfig)?;
        let prefix_head_start = head
            .checked_mul(seq_len)
            .and_then(|value| value.checked_mul(head_state_len))
            .ok_or(TrainCoreError::InvalidConfig)?;
        let denom_head_start = head
            .checked_mul(seq_len)
            .ok_or(TrainCoreError::InvalidConfig)?;
        workspace.linear_grad_state_q15[..head_state_len].fill(0);
        let state_start = head
            .checked_mul(head_state_len)
            .ok_or(TrainCoreError::InvalidConfig)?;
        let state_end = state_start
            .checked_add(head_state_len)
            .ok_or(TrainCoreError::InvalidConfig)?;
        let key_sum_start = head
            .checked_mul(head_dim)
            .ok_or(TrainCoreError::InvalidConfig)?;
        let key_sum_end = key_sum_start
            .checked_add(head_dim)
            .ok_or(TrainCoreError::InvalidConfig)?;
        workspace.attention_state_kv[state_start..state_end].fill(0);
        workspace.attention_key_sums[key_sum_start..key_sum_end].fill(0);

        for token in 0..seq_len {
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainCoreError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainCoreError::InvalidConfig)?;
            let key = &workspace.attention_k[row_start..row_end];
            let value = &workspace.attention_v[row_start..row_end];
            let state = &mut workspace.attention_state_kv[state_start..state_end];
            let key_sums = &mut workspace.attention_key_sums[key_sum_start..key_sum_end];

            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                key_sums[key_index] = key_sums[key_index]
                    .checked_add(phi_key)
                    .ok_or(TrainCoreError::CoreRejected)?;
                let state_row_start = key_index
                    .checked_mul(head_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                for (value_index, &value_value) in value.iter().enumerate() {
                    let product = phi_key
                        .checked_mul(i64::from(value_value))
                        .ok_or(TrainCoreError::CoreRejected)?;
                    let state_index = state_row_start
                        .checked_add(value_index)
                        .ok_or(TrainCoreError::InvalidConfig)?;
                    state[state_index] = state[state_index]
                        .checked_add(product)
                        .ok_or(TrainCoreError::CoreRejected)?;
                }
            }

            let query = &workspace.attention_q[row_start..row_end];
            let mut denominator = 0_i64;
            for (&query_value, &key_sum) in query.iter().zip(key_sums.iter()) {
                let product = mini_transformer_linear_attention_phi_i64(query_value)
                    .checked_mul(key_sum)
                    .ok_or(TrainCoreError::CoreRejected)?;
                denominator = denominator
                    .checked_add(product)
                    .ok_or(TrainCoreError::CoreRejected)?;
            }
            if denominator <= 0 {
                return Err(TrainCoreError::CoreRejected);
            }
            workspace.linear_denominators[denom_head_start + token] = denominator;
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainCoreError::InvalidConfig)?,
                )
                .ok_or(TrainCoreError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainCoreError::InvalidConfig)?;
            workspace.linear_prefix_states[snapshot_start..snapshot_end].copy_from_slice(state);
        }

        for query_index in 0..seq_len {
            let token = seq_len - 1 - query_index;
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainCoreError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainCoreError::InvalidConfig)?;
            let query = &workspace.attention_q[row_start..row_end];
            let key = &workspace.attention_k[row_start..row_end];
            let value = &workspace.attention_v[row_start..row_end];
            let grad_row = &workspace.grad_attention_context[row_start..row_end];
            let denominator = workspace.linear_denominators[denom_head_start + token];
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainCoreError::InvalidConfig)?,
                )
                .ok_or(TrainCoreError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainCoreError::InvalidConfig)?;
            let prefix_state = &workspace.linear_prefix_states[snapshot_start..snapshot_end];

            for key_dim in 0..head_dim {
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                let mut grad_q_numerator = 0_i64;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainCoreError::InvalidConfig)?;
                    let product = i64::from(grad_value)
                        .checked_mul(prefix_state[state_index])
                        .ok_or(TrainCoreError::CoreRejected)?;
                    grad_q_numerator = grad_q_numerator
                        .checked_add(product)
                        .ok_or(TrainCoreError::CoreRejected)?;
                }
                let target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                workspace.linear_grad_q_acc[target] = workspace.linear_grad_q_acc[target]
                    .checked_add(round_ratio_i64(grad_q_numerator, denominator)?)
                    .ok_or(TrainCoreError::CoreRejected)?;
            }

            for (key_dim, &query_value) in query.iter().enumerate() {
                let phi_query = mini_transformer_linear_attention_phi_i64(query_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let product = i64::from(grad_value)
                        .checked_mul(phi_query)
                        .and_then(|value| value.checked_mul(1_i64 << Q15_SHIFT))
                        .ok_or(TrainCoreError::CoreRejected)?;
                    let state_grad = round_ratio_i64(product, denominator)?;
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainCoreError::InvalidConfig)?;
                    workspace.linear_grad_state_q15[state_index] = workspace.linear_grad_state_q15
                        [state_index]
                        .checked_add(state_grad)
                        .ok_or(TrainCoreError::CoreRejected)?;
                }
            }

            for (key_dim, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                let mut grad_key_value = 0_i64;
                for (value_dim, &value_value) in value.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainCoreError::InvalidConfig)?;
                    let state_grad = workspace.linear_grad_state_q15[state_index];
                    let grad_v_product = state_grad
                        .checked_mul(phi_key)
                        .ok_or(TrainCoreError::CoreRejected)?;
                    let v_target = row_start
                        .checked_add(value_dim)
                        .ok_or(TrainCoreError::InvalidConfig)?;
                    workspace.linear_grad_v_acc[v_target] = workspace.linear_grad_v_acc[v_target]
                        .checked_add(round_shift_rhu_i64(grad_v_product, Q15_SHIFT))
                        .ok_or(TrainCoreError::CoreRejected)?;

                    let grad_k_product = state_grad
                        .checked_mul(i64::from(value_value))
                        .ok_or(TrainCoreError::CoreRejected)?;
                    grad_key_value = grad_key_value
                        .checked_add(round_shift_rhu_i64(grad_k_product, Q15_SHIFT))
                        .ok_or(TrainCoreError::CoreRejected)?;
                }
                let k_target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainCoreError::InvalidConfig)?;
                workspace.linear_grad_k_acc[k_target] = workspace.linear_grad_k_acc[k_target]
                    .checked_add(grad_key_value)
                    .ok_or(TrainCoreError::CoreRejected)?;
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

fn apply_mini_transformer_embedding_update_nope(
    embeddings: &mut [i16],
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainCoreError> {
    let total = context
        .len()
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainCoreError::InvalidConfig)?;
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || grad_embedding_output_q15.len() != total
        || learning_rate <= 0
        || embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainCoreError::InvalidConfig);
    }
    let mut stats = SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    };
    for (position, &token) in context.iter().enumerate() {
        let embedding_row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
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
                stats.zero_delta_count = stats.zero_delta_count.saturating_add(1);
            }
            apply_embedding_delta_i16(
                &mut embeddings[embedding_row_start + dim],
                delta,
                &mut stats,
            );
        }
    }
    Ok(stats)
}

fn apply_embedding_delta_i16(embedding: &mut i16, delta: i64, stats: &mut SoftmaxUpdateStats) {
    let previous = *embedding;
    let unclamped = i64::from(previous).saturating_add(delta);
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

fn add_i16_residual_rows_checked(
    left: &[i16],
    right: &[i16],
    output: &mut [i16],
) -> Result<usize, TrainCoreError> {
    if left.len() != right.len() || left.len() != output.len() {
        return Err(TrainCoreError::InvalidShape);
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

fn add_linear_weight_update_stats(
    left: LinearWeightUpdateStats,
    right: LinearWeightUpdateStats,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: left
            .gradient_saturation_count
            .saturating_add(right.gradient_saturation_count),
        zero_delta_count: left.zero_delta_count.saturating_add(right.zero_delta_count),
        weight_delta_l1: left.weight_delta_l1.saturating_add(right.weight_delta_l1),
    }
}

fn empty_linear_weight_update_stats() -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    }
}

fn byte_argmax_i32(logits: &[i32]) -> Result<u8, TrainCoreError> {
    if logits.len() != BYTE_VOCAB {
        return Err(TrainCoreError::InvalidShape);
    }
    Ok(logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, Reverse(index)))
        .map(|(index, _)| index as u8)
        .unwrap_or(0))
}

fn mini_transformer_head_dim() -> Option<usize> {
    if MINI_TRANSFORMER_HEADS == 0
        || MINI_TRANSFORMER_D_MODEL == 0
        || !MINI_TRANSFORMER_D_MODEL.is_multiple_of(MINI_TRANSFORMER_HEADS)
    {
        return None;
    }
    Some(MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS)
}

fn mini_transformer_linear_attention_phi_i64(value: i16) -> i64 {
    i64::from(value) + 32769
}

fn round_ratio_i64(numerator: i64, denominator: i64) -> Result<i64, TrainCoreError> {
    if denominator <= 0 {
        return Err(TrainCoreError::InvalidConfig);
    }
    let half = denominator / 2;
    if numerator >= 0 {
        numerator
            .checked_add(half)
            .map(|value| value / denominator)
            .ok_or(TrainCoreError::CoreRejected)
    } else {
        numerator
            .checked_neg()
            .and_then(|value| value.checked_add(half))
            .map(|value| -(value / denominator))
            .ok_or(TrainCoreError::CoreRejected)
    }
}

fn round_div_i64(numerator: i64, denominator: usize) -> Result<i64, TrainCoreError> {
    if denominator == 0 {
        return Err(TrainCoreError::InvalidConfig);
    }
    let denominator = i64::try_from(denominator).map_err(|_| TrainCoreError::InvalidConfig)?;
    round_ratio_i64(numerator, denominator)
}

fn rounded_shift_residual_i64(
    value: i64,
    shifted: i64,
    right_shift: u8,
) -> Result<i64, TrainCoreError> {
    if right_shift == 0 {
        return Ok(0);
    }

    let applied = i128::from(shifted)
        .checked_shl(u32::from(right_shift))
        .ok_or(TrainCoreError::CoreRejected)?;
    let residual = i128::from(value)
        .checked_sub(applied)
        .ok_or(TrainCoreError::CoreRejected)?;
    i64::try_from(residual).map_err(|_| TrainCoreError::CoreRejected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn rejects_bad_workspace_shape() {
        let mut embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let mut weights = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL];
        let mut up = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM];
        let mut down = vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL];
        let mut output = vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
        let mut model = MiniTransformerModelSlicesMut {
            embeddings: &mut embeddings,
            q_weights: &mut weights,
            k_weights: &mut [],
            v_weights: &mut [],
            o_weights: &mut [],
            up_weights: &mut up,
            gate_weights: &mut [],
            down_weights: &mut down,
            output_weights: &mut output,
        };
        let mut empty_i16 = [];
        let mut empty_i32 = [];
        let mut empty_i64 = [];
        let mut workspace = MiniTransformerStepWorkspace {
            embedding_output: &mut empty_i16,
            attention_norm: &mut [],
            attention_q: &mut [],
            attention_k: &mut [],
            attention_v: &mut [],
            attention_context: &mut [],
            attention_output: &mut [],
            attention_residual: &mut [],
            attention_state_kv: &mut empty_i64,
            attention_key_sums: &mut [],
            mlp_norm: &mut [],
            mlp_up: &mut [],
            mlp_gate: &mut [],
            mlp_gated: &mut [],
            mlp_output: &mut [],
            block_output: &mut [],
            logits_q8: &mut empty_i32,
            probabilities_q15: &mut [],
            grad_output_q15: &mut [],
            output_scaled_grad: &mut [],
            grad_last_features: &mut [],
            grad_mlp_output: &mut [],
            grad_mlp_input: &mut [],
            mlp_scaled_grad: &mut [],
            mlp_input_grad_gated: &mut [],
            mlp_input_grad_up: &mut [],
            mlp_input_grad_gate: &mut [],
            mlp_input_grad_up_input: &mut [],
            mlp_input_grad_gate_input: &mut [],
            mlp_update_grad_gated: &mut [],
            mlp_update_grad_up: &mut [],
            mlp_update_grad_gate: &mut [],
            grad_attention_output: &mut [],
            grad_attention_context: &mut [],
            attention_scaled_grad: &mut [],
            linear_prefix_states: &mut [],
            linear_denominators: &mut [],
            linear_grad_state_q15: &mut [],
            linear_grad_q_acc: &mut [],
            linear_grad_k_acc: &mut [],
            linear_grad_v_acc: &mut [],
            grad_attention_q: &mut [],
            grad_attention_k: &mut [],
            grad_attention_v: &mut [],
            grad_attention_norm_input: &mut [],
            grad_embedding_output: &mut [],
        };
        let err = mini_transformer_linear_nope_train_step(
            &mut model,
            &[1, 2, 3, 4],
            5,
            MiniTransformerStepConfig {
                seq_len: 4,
                learning_rate: 1,
                output_learning_rate_shift: 18,
                mlp_learning_rate_shift: 17,
                embedding_learning_rate_shift: 13,
                attention_learning_rate_shift: 22,
                attention_q_learning_rate_shift: 18,
                attention_qk_learning_rate_shift: 16,
            },
            &mut workspace,
        )
        .unwrap_err();
        assert_eq!(err, TrainCoreError::InvalidShape);
    }

    #[test]
    fn output_scale_table_matches_the_forward_scale() {
        assert!(
            MINI_TRANSFORMER_OUTPUT_SCALES
                .iter()
                .all(|&scale| scale == MINI_TRANSFORMER_DEFAULT_OUTPUT_SCALE)
        );
    }

    #[test]
    fn linear_weight_gradient_i64_averages_then_updates_i8() {
        let mut accumulators = vec![0_i64; 4];
        let mut residuals = vec![0_i64; 4];
        let mut gradient = LinearWeightGradientI64Workspace {
            input_dim: 2,
            output_dim: 2,
            sample_count: 0,
            accumulators: &mut accumulators,
            residuals: &mut residuals,
        };
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
        let mut accumulators = vec![0_i64; 1];
        let mut residuals = vec![0_i64; 1];
        let mut gradient = LinearWeightGradientI64Workspace {
            input_dim: 1,
            output_dim: 1,
            sample_count: 0,
            accumulators: &mut accumulators,
            residuals: &mut residuals,
        };
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
    fn integer_adam_zero_gradient_keeps_weights_and_advances_state() {
        let accumulators = [0_i64; 2];
        let mut weights = [7_i8, -9_i8];
        let mut first = [0_i64; 2];
        let mut second = [0_u64; 2];
        let mut residuals = [0_i64; 2];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };

        let stats = apply_integer_adam_accumulators_i64_to_i8(
            &accumulators,
            1,
            &mut weights,
            IntegerAdamConfig::default(),
            &mut state,
        )
        .expect("zero gradient");

        assert_eq!(weights, [7, -9]);
        assert_eq!(state.step, 1);
        assert_eq!(state.first_moments, [0, 0]);
        assert_eq!(state.second_moments, [0, 0]);
        assert_eq!(stats.zero_delta_count, 2);
        assert_eq!(stats.weight_delta_l1, 0);
    }

    #[test]
    fn integer_adam_constant_gradient_moves_in_descent_direction() {
        let accumulators = [1024_i64, -1024_i64];
        let mut weights = [10_i8, -10_i8];
        let mut first = [0_i64; 2];
        let mut second = [0_u64; 2];
        let mut residuals = [0_i64; 2];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };
        let config = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };

        for _ in 0..4 {
            apply_integer_adam_accumulators_i64_to_i8(
                &accumulators,
                1,
                &mut weights,
                config,
                &mut state,
            )
            .expect("constant gradient");
        }

        assert!(weights[0] < 10);
        assert!(weights[1] > -10);
        assert_eq!(state.step, 4);
        assert!(state.first_moments[0] > 0 && state.first_moments[1] < 0);
        assert!(state.second_moments.iter().all(|&value| value > 0));
    }

    #[test]
    fn integer_adam_carries_sub_i8_updates() {
        let accumulators = [4096_i64];
        let mut weights = [10_i8];
        let mut first = [0_i64];
        let mut second = [0_u64];
        let mut residuals = [0_i64];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };
        let config = IntegerAdamConfig {
            step_shift: 4,
            ..IntegerAdamConfig::default()
        };

        let first_stats = apply_integer_adam_accumulators_i64_to_i8(
            &accumulators,
            1,
            &mut weights,
            config,
            &mut state,
        )
        .expect("first subthreshold update");
        assert_eq!(weights, [10]);
        assert_eq!(first_stats.zero_delta_count, 1);
        assert_ne!(state.update_residuals, [0]);

        for _ in 0..20 {
            apply_integer_adam_accumulators_i64_to_i8(
                &accumulators,
                1,
                &mut weights,
                config,
                &mut state,
            )
            .expect("carried update");
        }
        assert!(weights[0] < 10);
    }

    #[test]
    fn integer_adam_reports_gradient_and_weight_saturation() {
        let accumulators = [i64::MAX];
        let mut weights = [i8::MIN];
        let mut first = [0_i64];
        let mut second = [0_u64];
        let mut residuals = [0_i64];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };
        let config = IntegerAdamConfig {
            step_shift: 0,
            ..IntegerAdamConfig::default()
        };

        let stats = apply_integer_adam_accumulators_i64_to_i8(
            &accumulators,
            1,
            &mut weights,
            config,
            &mut state,
        )
        .expect("saturated update");

        assert_eq!(weights, [i8::MIN]);
        assert!(stats.gradient_saturation_count >= 2);
        assert_eq!(state.update_residuals, [0]);
    }

    #[test]
    fn integer_adam_i16_path_is_deterministic() {
        fn run() -> (std::vec::Vec<i16>, std::vec::Vec<i64>, std::vec::Vec<u64>) {
            let accumulators = [500_i64, -750_i64, 125_i64];
            let mut weights = vec![100_i16, -100_i16, 0_i16];
            let mut first = vec![0_i64; 3];
            let mut second = vec![0_u64; 3];
            let mut residuals = vec![0_i64; 3];
            let mut state = IntegerAdamStateWorkspace {
                step: 0,
                first_moments: &mut first,
                second_moments: &mut second,
                update_residuals: &mut residuals,
            };
            for _ in 0..12 {
                apply_integer_adam_accumulators_i64_to_i16(
                    &accumulators,
                    2,
                    &mut weights,
                    IntegerAdamConfig::default(),
                    &mut state,
                )
                .expect("i16 update");
            }
            (weights, first, second)
        }

        assert_eq!(run(), run());
    }

    #[test]
    fn integer_adam_rejects_invalid_config_and_state_shape() {
        let accumulators = [1_i64];
        let mut weights = [0_i8];
        let mut first = [];
        let mut second = [];
        let mut residuals = [];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };
        assert_eq!(
            apply_integer_adam_accumulators_i64_to_i8(
                &accumulators,
                1,
                &mut weights,
                IntegerAdamConfig::default(),
                &mut state,
            ),
            Err(TrainCoreError::InvalidShape)
        );

        let mut first = [0_i64];
        let mut second = [0_u64];
        let mut residuals = [0_i64];
        let mut state = IntegerAdamStateWorkspace {
            step: 0,
            first_moments: &mut first,
            second_moments: &mut second,
            update_residuals: &mut residuals,
        };
        assert_eq!(
            apply_integer_adam_accumulators_i64_to_i8(
                &accumulators,
                1,
                &mut weights,
                IntegerAdamConfig {
                    epsilon: 0,
                    ..IntegerAdamConfig::default()
                },
                &mut state,
            ),
            Err(TrainCoreError::InvalidConfig)
        );
    }
}
