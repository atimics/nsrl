use crate::numeric::{
    FixedScale, MAX_RIGHT_SHIFT, requantize_i32_to_i16, round_shift_rhu_i64, saturate_i8,
    saturate_i16,
};

#[derive(Debug, Clone, Copy)]
pub struct LinearI16I8Params<'a> {
    pub weights: &'a [i8],
    pub bias: Option<&'a [i32]>,
    pub scales: &'a [FixedScale],
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearKernel {
    GenericI8,
    Ternary,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearBackwardInputI16I8Params<'a> {
    pub weights: &'a [i8],
    pub forward_scales: &'a [FixedScale],
    pub grad_input_scales: &'a [FixedScale],
    pub input_dim: usize,
    pub output_dim: usize,
}

pub struct LinearBackwardInputWorkspace<'a> {
    pub scaled_grad_output: &'a mut [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct LinearBackwardWeightUpdateI8Params<'a> {
    pub forward_scales: &'a [FixedScale],
    pub input_dim: usize,
    pub output_dim: usize,
    pub learning_rate: i32,
    pub learning_rate_shift: u8,
}

pub struct LinearBackwardWeightUpdateWorkspace<'a> {
    pub scaled_grad_output: &'a mut [i32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearWeightUpdateStats {
    pub gradient_saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
}

impl LinearI16I8Params<'_> {
    pub fn is_valid(self) -> bool {
        if self.input_dim == 0 || self.output_dim == 0 {
            return false;
        }

        let Some(expected_weights) = self.input_dim.checked_mul(self.output_dim) else {
            return false;
        };

        if self.weights.len() != expected_weights || self.scales.len() != self.output_dim {
            return false;
        }

        if let Some(bias) = self.bias
            && bias.len() != self.output_dim
        {
            return false;
        }

        true
    }
}

impl LinearBackwardInputI16I8Params<'_> {
    pub fn is_valid(self) -> bool {
        if self.input_dim == 0 || self.output_dim == 0 {
            return false;
        }

        let Some(expected_weights) = self.input_dim.checked_mul(self.output_dim) else {
            return false;
        };

        self.weights.len() == expected_weights
            && self.forward_scales.len() == self.output_dim
            && self.grad_input_scales.len() == self.input_dim
            && self
                .forward_scales
                .iter()
                .chain(self.grad_input_scales.iter())
                .all(|scale| scale.multiplier >= 0 && scale.right_shift <= MAX_RIGHT_SHIFT)
    }
}

impl LinearBackwardWeightUpdateI8Params<'_> {
    pub fn is_valid(self) -> bool {
        self.input_dim != 0
            && self.output_dim != 0
            && self.learning_rate > 0
            && self.learning_rate_shift <= MAX_RIGHT_SHIFT
            && self.forward_scales.len() == self.output_dim
            && self
                .forward_scales
                .iter()
                .all(|scale| scale.multiplier >= 0 && scale.right_shift <= MAX_RIGHT_SHIFT)
    }
}

pub fn linear_i16_i8_i16_per_channel_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
) -> Option<()> {
    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input,
        params,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn linear_i16_i8_i16_per_channel_with_kernel_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
    kernel: LinearKernel,
) -> Option<()> {
    match kernel {
        LinearKernel::GenericI8 => {
            linear_i16_i8_i16_per_channel_generic_checked(input, params, output)
        }
        LinearKernel::Ternary => linear_i16_ternary_i16_per_channel_checked(input, params, output),
    }
}

pub fn linear_i16_i8_i16_per_channel_generic_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
) -> Option<()> {
    if input.len() != params.input_dim || output.len() != params.output_dim || !params.is_valid() {
        return None;
    }

    for (out_index, out) in output.iter_mut().enumerate() {
        let row_start = out_index.checked_mul(params.input_dim)?;
        let row_end = row_start.checked_add(params.input_dim)?;
        let weights = &params.weights[row_start..row_end];
        let mut acc = params.bias.map_or(0_i32, |bias| bias[out_index]);

        for (&activation, &weight) in input.iter().zip(weights.iter()) {
            let product = i32::from(activation) * i32::from(weight);
            acc = acc.checked_add(product)?;
        }

        *out = requantize_i32_to_i16(acc, params.scales[out_index]);
    }

    Some(())
}

pub fn linear_i16_ternary_i16_per_channel_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
) -> Option<()> {
    if input.len() != params.input_dim || output.len() != params.output_dim || !params.is_valid() {
        return None;
    }

    for (out_index, out) in output.iter_mut().enumerate() {
        let row_start = out_index.checked_mul(params.input_dim)?;
        let row_end = row_start.checked_add(params.input_dim)?;
        let weights = &params.weights[row_start..row_end];
        let mut acc = params.bias.map_or(0_i32, |bias| bias[out_index]);

        for (&activation, &weight) in input.iter().zip(weights.iter()) {
            match weight {
                -1 => acc = acc.checked_sub(i32::from(activation))?,
                0 => {}
                1 => acc = acc.checked_add(i32::from(activation))?,
                _ => return None,
            }
        }

        *out = requantize_i32_to_i16(acc, params.scales[out_index]);
    }

    Some(())
}

pub fn linear_backward_input_i16_i8_i16_per_channel_checked(
    grad_output: &[i16],
    params: LinearBackwardInputI16I8Params<'_>,
    workspace: LinearBackwardInputWorkspace<'_>,
    grad_input: &mut [i16],
) -> Option<()> {
    if grad_output.len() != params.output_dim
        || grad_input.len() != params.input_dim
        || workspace.scaled_grad_output.len() != params.output_dim
        || !params.is_valid()
    {
        return None;
    }

    linear_backward_prescale_grad_output_i16_i32_checked(
        grad_output,
        params.forward_scales,
        workspace.scaled_grad_output,
    )?;

    linear_backward_input_prescaled_i32_i8_i16_per_channel_checked(
        workspace.scaled_grad_output,
        params.weights,
        params.input_dim,
        params.output_dim,
        params.grad_input_scales,
        grad_input,
    )
}

pub fn linear_backward_prescale_grad_output_i16_i32_checked(
    grad_output: &[i16],
    forward_scales: &[FixedScale],
    scaled_grad_output: &mut [i32],
) -> Option<()> {
    if grad_output.len() != forward_scales.len() || grad_output.len() != scaled_grad_output.len() {
        return None;
    }

    for ((&grad, &scale), out) in grad_output
        .iter()
        .zip(forward_scales.iter())
        .zip(scaled_grad_output.iter_mut())
    {
        let scaled = scale_i16_by_fixed_scale_to_i64(grad, scale)?;
        *out = i32::try_from(scaled).ok()?;
    }

    Some(())
}

pub fn linear_backward_input_prescaled_i32_i8_i16_per_channel_checked(
    scaled_grad_output: &[i32],
    weights: &[i8],
    input_dim: usize,
    output_dim: usize,
    grad_input_scales: &[FixedScale],
    grad_input: &mut [i16],
) -> Option<()> {
    if input_dim == 0
        || output_dim == 0
        || scaled_grad_output.len() != output_dim
        || grad_input.len() != input_dim
        || grad_input_scales.len() != input_dim
        || weights.len() != input_dim.checked_mul(output_dim)?
        || grad_input_scales
            .iter()
            .any(|scale| scale.multiplier < 0 || scale.right_shift > MAX_RIGHT_SHIFT)
    {
        return None;
    }

    for (in_index, out) in grad_input.iter_mut().enumerate() {
        let mut acc = 0_i64;

        for (out_index, &scaled_grad) in scaled_grad_output.iter().enumerate() {
            let weight_index = out_index.checked_mul(input_dim)?.checked_add(in_index)?;
            let product = i64::from(scaled_grad).checked_mul(i64::from(weights[weight_index]))?;
            acc = acc.checked_add(product)?;
        }

        *out = requantize_i64_to_i16(acc, grad_input_scales[in_index])?;
    }

    Some(())
}

pub fn linear_backward_weight_update_i8_checked(
    input: &[i16],
    grad_output: &[i16],
    weights: &mut [i8],
    params: LinearBackwardWeightUpdateI8Params<'_>,
    workspace: LinearBackwardWeightUpdateWorkspace<'_>,
) -> Option<LinearWeightUpdateStats> {
    if input.len() != params.input_dim
        || grad_output.len() != params.output_dim
        || weights.len() != params.input_dim.checked_mul(params.output_dim)?
        || workspace.scaled_grad_output.len() != params.output_dim
        || !params.is_valid()
    {
        return None;
    }

    linear_backward_prescale_grad_output_i16_i32_checked(
        grad_output,
        params.forward_scales,
        workspace.scaled_grad_output,
    )?;

    linear_backward_weight_update_prescaled_i32_i8_checked(
        input,
        workspace.scaled_grad_output,
        weights,
        params.input_dim,
        params.output_dim,
        params.learning_rate,
        params.learning_rate_shift,
    )
}

pub fn linear_backward_weight_update_prescaled_i32_i8_checked(
    input: &[i16],
    scaled_grad_output: &[i32],
    weights: &mut [i8],
    input_dim: usize,
    output_dim: usize,
    learning_rate: i32,
    learning_rate_shift: u8,
) -> Option<LinearWeightUpdateStats> {
    if input_dim == 0
        || output_dim == 0
        || input.len() != input_dim
        || scaled_grad_output.len() != output_dim
        || weights.len() != input_dim.checked_mul(output_dim)?
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return None;
    }

    let mut stats = LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    };

    for (out_index, &scaled_grad) in scaled_grad_output.iter().enumerate() {
        let row_start = out_index.checked_mul(input_dim)?;
        for (in_index, &activation) in input.iter().enumerate() {
            if activation == 0 || scaled_grad == 0 {
                continue;
            }

            let product = i64::from(scaled_grad)
                .checked_mul(i64::from(activation))?
                .checked_mul(i64::from(learning_rate))?;
            let scaled_update = round_shift_rhu_i64(product, learning_rate_shift);
            let delta = -scaled_update;
            if delta == 0 {
                stats.zero_delta_count = stats.zero_delta_count.checked_add(1)?;
            }

            let weight = &mut weights[row_start.checked_add(in_index)?];
            let previous = *weight;
            let unclamped = i64::from(previous).checked_add(delta)?;
            let clamped = saturate_i8(unclamped);
            if i64::from(clamped) != unclamped {
                stats.gradient_saturation_count = stats.gradient_saturation_count.checked_add(1)?;
            }
            let applied_delta = i64::from(clamped) - i64::from(previous);
            stats.weight_delta_l1 = stats
                .weight_delta_l1
                .checked_add(applied_delta.unsigned_abs())?;
            *weight = clamped;
        }
    }

    Some(stats)
}

fn scale_i16_by_fixed_scale_to_i64(value: i16, scale: FixedScale) -> Option<i64> {
    if scale.multiplier < 0 || scale.right_shift > MAX_RIGHT_SHIFT {
        return None;
    }

    let wide = i64::from(value).checked_mul(i64::from(scale.multiplier))?;
    Some(round_shift_rhu_i64(wide, scale.right_shift))
}

fn requantize_i64_to_i16(accumulator: i64, scale: FixedScale) -> Option<i16> {
    if scale.multiplier < 0 || scale.right_shift > MAX_RIGHT_SHIFT {
        return None;
    }

    let wide = accumulator.checked_mul(i64::from(scale.multiplier))?;
    Some(saturate_i16(round_shift_rhu_i64(wide, scale.right_shift)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RESIDUAL_Q15_SCALE;

    #[test]
    fn linear_identity_projection_preserves_input() {
        let input = [100_i16, -200, 300, -400];
        let weights = [1_i8, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let scales = [RESIDUAL_Q15_SCALE; 4];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 4,
            output_dim: 4,
        };
        let mut output = [0_i16; 4];

        assert!(linear_i16_i8_i16_per_channel_checked(&input, params, &mut output).is_some());
        assert_eq!(output, input);
    }

    #[test]
    fn linear_applies_bias_and_per_channel_scale() {
        let input = [100_i16, -50];
        let weights = [2_i8, -1, 2, -1];
        let bias = [10_i32, 10];
        let scales = [
            FixedScale {
                multiplier: 1,
                right_shift: 0,
            },
            FixedScale {
                multiplier: 1,
                right_shift: 1,
            },
        ];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&bias),
            scales: &scales,
            input_dim: 2,
            output_dim: 2,
        };
        let mut output = [0_i16; 2];

        assert!(linear_i16_i8_i16_per_channel_checked(&input, params, &mut output).is_some());
        assert_eq!(output, [260, 130]);
    }

    #[test]
    fn ternary_linear_matches_generic_for_ternary_weights() {
        let input = [100_i16, -50, 25];
        let weights = [
            1_i8, 0, -1, //
            -1, 1, 0,
        ];
        let bias = [10_i32, -10];
        let scales = [RESIDUAL_Q15_SCALE; 2];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&bias),
            scales: &scales,
            input_dim: 3,
            output_dim: 2,
        };
        let mut generic = [0_i16; 2];
        let mut ternary = [0_i16; 2];

        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut generic).is_some()
        );
        assert!(linear_i16_ternary_i16_per_channel_checked(&input, params, &mut ternary).is_some());
        assert_eq!(generic, ternary);
    }

    #[test]
    fn ternary_linear_rejects_non_ternary_weights() {
        let input = [100_i16, -50];
        let weights = [2_i8, -1];
        let scales = [RESIDUAL_Q15_SCALE; 1];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 2,
            output_dim: 1,
        };
        let mut output = [0_i16; 1];

        assert_eq!(
            linear_i16_ternary_i16_per_channel_checked(&input, params, &mut output),
            None
        );
    }

    #[test]
    fn linear_rejects_bad_shapes() {
        let weights = [1_i8; 4];
        let scales = [RESIDUAL_Q15_SCALE; 2];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 2,
            output_dim: 2,
        };
        let mut output = [0_i16; 3];

        assert_eq!(
            linear_i16_i8_i16_per_channel_checked(&[1_i16, 2], params, &mut output),
            None
        );
    }

    #[test]
    fn linear_rejects_accumulator_overflow() {
        let input = [i16::MAX; 600];
        let weights = [127_i8; 600];
        let scales = [RESIDUAL_Q15_SCALE; 1];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&[i32::MAX]),
            scales: &scales,
            input_dim: 600,
            output_dim: 1,
        };
        let mut output = [0_i16; 1];

        assert_eq!(
            linear_i16_i8_i16_per_channel_checked(&input, params, &mut output),
            None
        );
    }

    #[test]
    fn linear_backward_input_transposes_weight_matrix() {
        let grad_output = [100_i16, -50, 25];
        let weights = [
            2_i8, -1, //
            3, 4, //
            -2, 5,
        ];
        let forward_scales = [RESIDUAL_Q15_SCALE; 3];
        let grad_input_scales = [RESIDUAL_Q15_SCALE; 2];
        let params = LinearBackwardInputI16I8Params {
            weights: &weights,
            forward_scales: &forward_scales,
            grad_input_scales: &grad_input_scales,
            input_dim: 2,
            output_dim: 3,
        };
        let mut grad_input = [0_i16; 2];
        let mut scaled_grad_output = [0_i32; 3];

        assert!(
            linear_backward_input_i16_i8_i16_per_channel_checked(
                &grad_output,
                params,
                LinearBackwardInputWorkspace {
                    scaled_grad_output: &mut scaled_grad_output,
                },
                &mut grad_input,
            )
            .is_some()
        );
        assert_eq!(scaled_grad_output, [100, -50, 25]);
        assert_eq!(grad_input, [0, -175]);
    }

    #[test]
    fn linear_backward_input_applies_forward_and_input_scales() {
        let grad_output = [100_i16, 100];
        let weights = [
            2_i8, 2, //
            2, 2,
        ];
        let forward_scales = [
            FixedScale {
                multiplier: 1,
                right_shift: 1,
            },
            RESIDUAL_Q15_SCALE,
        ];
        let grad_input_scales = [
            FixedScale {
                multiplier: 1,
                right_shift: 1,
            },
            RESIDUAL_Q15_SCALE,
        ];
        let params = LinearBackwardInputI16I8Params {
            weights: &weights,
            forward_scales: &forward_scales,
            grad_input_scales: &grad_input_scales,
            input_dim: 2,
            output_dim: 2,
        };
        let mut grad_input = [0_i16; 2];
        let mut scaled_grad_output = [0_i32; 2];

        assert!(
            linear_backward_input_i16_i8_i16_per_channel_checked(
                &grad_output,
                params,
                LinearBackwardInputWorkspace {
                    scaled_grad_output: &mut scaled_grad_output,
                },
                &mut grad_input,
            )
            .is_some()
        );
        assert_eq!(scaled_grad_output, [50, 100]);
        assert_eq!(grad_input, [150, 300]);
    }

    #[test]
    fn linear_backward_weight_update_uses_prescaled_outer_product() {
        let input = [64_i16, -32];
        let grad_output = [32_i16, -64];
        let mut weights = [0_i8; 4];
        let forward_scales = [
            RESIDUAL_Q15_SCALE,
            FixedScale {
                multiplier: 1,
                right_shift: 1,
            },
        ];
        let params = LinearBackwardWeightUpdateI8Params {
            forward_scales: &forward_scales,
            input_dim: 2,
            output_dim: 2,
            learning_rate: 1,
            learning_rate_shift: 10,
        };
        let mut scaled_grad_output = [0_i32; 2];

        let stats = linear_backward_weight_update_i8_checked(
            &input,
            &grad_output,
            &mut weights,
            params,
            LinearBackwardWeightUpdateWorkspace {
                scaled_grad_output: &mut scaled_grad_output,
            },
        )
        .expect("update");

        assert_eq!(scaled_grad_output, [32, -32]);
        assert_eq!(weights, [-2, 1, 2, -1]);
        assert_eq!(
            stats,
            LinearWeightUpdateStats {
                gradient_saturation_count: 0,
                zero_delta_count: 0,
                weight_delta_l1: 6,
            }
        );
    }

    #[test]
    fn linear_backward_weight_update_reports_zero_delta_and_saturation() {
        let input = [1_i16, 512];
        let scaled_grad_output = [1_i32, -1024];
        let mut weights = [0_i8, i8::MAX, 0, i8::MIN];

        let stats = linear_backward_weight_update_prescaled_i32_i8_checked(
            &input,
            &scaled_grad_output,
            &mut weights,
            2,
            2,
            1,
            10,
        )
        .expect("update");

        assert_eq!(weights, [0, 126, 1, 127]);
        assert_eq!(stats.zero_delta_count, 1);
        assert_eq!(stats.gradient_saturation_count, 1);
        assert_eq!(stats.weight_delta_l1, 257);
    }
}
