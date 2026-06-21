#![no_std]
#![deny(unsafe_code)]

#[cfg(test)]
extern crate std;

pub mod attention;
pub mod linear;
pub mod lut;
pub mod mlp;
pub mod numeric;
pub mod rms_norm;
pub mod tensor;

pub use attention::{
    AttentionResidualWorkspace, LOGIT_FRAC_BITS, LinearAttentionState,
    LinearAttentionStepWorkspace, LinearAttentionTttStepWorkspace, LinearAttentionWorkspace,
    MASKED_LOGIT, PreNormAttentionResidualWorkspace, Q15_SHIFT, SelfAttentionI16Params,
    SelfAttentionWorkspace, attention_dot_q_k_i16_i32_checked,
    attention_residual_block_i16_q15_checked,
    attention_residual_block_i16_q15_with_linear_kernel_checked, attention_row_i16_q15_checked,
    attention_weight_v_i16_q15_checked, base2_exp_neg_q15, base2_softmax_i32_q15,
    clear_linear_attention_state_checked, is_power_of_four, linear_attention_i16_q15_checked,
    linear_attention_i16_q15_with_linear_kernel_checked, linear_attention_state_lengths,
    linear_attention_step_i16_q15_checked,
    linear_attention_step_i16_q15_with_linear_kernel_checked,
    linear_attention_ttt_delta_state_i16_q15_checked, linear_attention_ttt_step_i16_q15_checked,
    linear_attention_ttt_step_i16_q15_with_linear_kernel_checked,
    prenorm_attention_residual_block_i16_q15_checked,
    prenorm_attention_residual_block_i16_q15_with_linear_kernel_checked, reciprocal_sum_q31,
    self_attention_i16_q15_checked, self_attention_i16_q15_with_linear_kernel_checked,
    sqrt_power_of_four_shift,
};

pub use linear::{
    LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace,
    LinearBackwardWeightUpdateI8Params, LinearBackwardWeightUpdateWorkspace, LinearI16I8Params,
    LinearKernel, LinearWeightUpdateStats, linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_backward_input_prescaled_i32_i8_i16_per_channel_checked,
    linear_backward_prescale_grad_output_i16_i32_checked, linear_backward_weight_update_i8_checked,
    linear_backward_weight_update_prescaled_i32_i8_checked, linear_i16_i8_i16_per_channel_checked,
    linear_i16_i8_i16_per_channel_generic_checked,
    linear_i16_i8_i16_per_channel_with_kernel_checked, linear_i16_ternary_i16_per_channel_checked,
};
pub use lut::{
    EXP2_NEG_FRAC_LUT_8BIT_LEN, NormalizedMantissa, RECIP_LUT_8BIT_Q31_LEN,
    RSQRT_LUT_8BIT_DOMAIN_END, RSQRT_LUT_8BIT_DOMAIN_START, RSQRT_LUT_8BIT_LEN,
    normalize_u64_to_lut_index, recip_lut_8bit_q31, rsqrt_lut_8bit_q15,
};
pub use mlp::{
    GatedActivationGradQ15, GatedMlpBackwardScales, GatedMlpBackwardWorkspace, GatedMlpI16Params,
    GatedMlpResidualWorkspace, GatedMlpWeightUpdateParams, GatedMlpWeightUpdateStats,
    GatedMlpWeightUpdateWorkspace, GatedMlpWorkspace, HARD_SILU_GATE_BIAS_Q15,
    HARD_SILU_GATE_SHIFT, PreNormGatedMlpResidualWorkspace, gated_activation_backward_i16_q15,
    gated_activation_i16_q15, gated_mlp_backward_input_i16_q15_checked,
    gated_mlp_backward_weight_update_i8_checked, gated_mlp_i16_q15_checked,
    gated_mlp_i16_q15_with_linear_kernel_checked, gated_mlp_residual_block_i16_q15_checked,
    gated_mlp_residual_block_i16_q15_with_linear_kernel_checked, hard_silu_derivative_q15,
    hard_silu_gate_q15, hard_silu_q15, prenorm_gated_mlp_residual_block_i16_q15_checked,
    prenorm_gated_mlp_residual_block_i16_q15_with_linear_kernel_checked,
};
pub use numeric::{
    FixedScale, MAX_RIGHT_SHIFT, RESIDUAL_Q15_SCALE, dot_i8_i16_i32_checked, requantize_i32_to_i16,
    residual_add_i16_q15_checked, round_shift_rhu_i64, saturate_i8, saturate_i16,
    saturating_add_i16,
};
pub use rms_norm::{
    INV_SQRT_2_Q15, RMSNORM_INV_RMS_SHIFT, integer_rsqrt_q30, rms_norm_i16_q15_checked,
    sum_squares_i16_u64_checked,
};
pub use tensor::TensorView;
