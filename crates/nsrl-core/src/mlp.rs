use crate::linear::{
    LinearBackwardInputI16I8Params, LinearBackwardInputWorkspace, LinearI16I8Params, LinearKernel,
    linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_i16_i8_i16_per_channel_with_kernel_checked,
};
use crate::numeric::{FixedScale, round_shift_rhu_i64, saturate_i16, saturating_add_i16};
use crate::rms_norm::rms_norm_i16_q15_checked;

pub const HARD_SILU_GATE_SHIFT: u8 = 2;
pub const HARD_SILU_GATE_BIAS_Q15: i32 = 16_384;

#[derive(Debug, Clone, Copy)]
pub struct GatedMlpI16Params<'a> {
    pub up: LinearI16I8Params<'a>,
    pub gate: LinearI16I8Params<'a>,
    pub down: LinearI16I8Params<'a>,
    pub seq_len: usize,
    pub d_model: usize,
    pub hidden_dim: usize,
}

pub struct GatedMlpWorkspace<'a> {
    pub up: &'a mut [i16],
    pub gate: &'a mut [i16],
    pub gated: &'a mut [i16],
}

pub struct GatedMlpResidualWorkspace<'a> {
    pub mlp: GatedMlpWorkspace<'a>,
    pub mlp_output: &'a mut [i16],
}

pub struct PreNormGatedMlpResidualWorkspace<'a> {
    pub normalized: &'a mut [i16],
    pub residual: GatedMlpResidualWorkspace<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct GatedMlpBackwardScales<'a> {
    pub down_to_hidden: &'a [FixedScale],
    pub up_to_input: &'a [FixedScale],
    pub gate_to_input: &'a [FixedScale],
}

pub struct GatedMlpBackwardWorkspace<'a> {
    pub scaled_grad_output: &'a mut [i32],
    pub grad_gated: &'a mut [i16],
    pub grad_up: &'a mut [i16],
    pub grad_gate: &'a mut [i16],
    pub grad_up_input: &'a mut [i16],
    pub grad_gate_input: &'a mut [i16],
}

impl GatedMlpI16Params<'_> {
    pub fn is_valid(self) -> bool {
        if self.seq_len == 0 || self.d_model == 0 || self.hidden_dim == 0 {
            return false;
        }

        self.up.input_dim == self.d_model
            && self.up.output_dim == self.hidden_dim
            && self.gate.input_dim == self.d_model
            && self.gate.output_dim == self.hidden_dim
            && self.down.input_dim == self.hidden_dim
            && self.down.output_dim == self.d_model
            && self.up.is_valid()
            && self.gate.is_valid()
            && self.down.is_valid()
    }
}

pub fn hard_silu_gate_q15(value: i16) -> i16 {
    let shifted = i32::from(value) >> HARD_SILU_GATE_SHIFT;
    let biased = shifted + HARD_SILU_GATE_BIAS_Q15;
    biased.clamp(0, i32::from(i16::MAX)) as i16
}

pub fn hard_silu_q15(value: i16) -> i16 {
    let gate_q15 = hard_silu_gate_q15(value);
    let product = i64::from(value) * i64::from(gate_q15);
    saturate_i16(round_shift_rhu_i64(product, 15))
}

pub fn hard_silu_derivative_q15(value: i16) -> i16 {
    ((i32::from(value) >> 1) + HARD_SILU_GATE_BIAS_Q15) as i16
}

pub fn gated_activation_i16_q15(up: i16, gate: i16) -> i16 {
    let silu_q15 = hard_silu_q15(gate);
    let product = i64::from(up) * i64::from(silu_q15);
    saturate_i16(round_shift_rhu_i64(product, 15))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedActivationGradQ15 {
    pub up: i16,
    pub gate: i16,
}

pub fn gated_activation_backward_i16_q15(
    up: i16,
    gate: i16,
    grad_output: i16,
) -> GatedActivationGradQ15 {
    let silu_q15 = hard_silu_q15(gate);
    let grad_up = round_shift_rhu_i64(i64::from(grad_output) * i64::from(silu_q15), 15);

    let grad_silu = round_shift_rhu_i64(i64::from(grad_output) * i64::from(up), 15);
    let derivative_q15 = hard_silu_derivative_q15(gate);
    let grad_gate = round_shift_rhu_i64(grad_silu * i64::from(derivative_q15), 15);

    GatedActivationGradQ15 {
        up: saturate_i16(grad_up),
        gate: saturate_i16(grad_gate),
    }
}

pub fn gated_mlp_i16_q15_checked(
    input: &[i16],
    params: GatedMlpI16Params<'_>,
    workspace: GatedMlpWorkspace<'_>,
    output: &mut [i16],
) -> Option<()> {
    gated_mlp_i16_q15_with_linear_kernel_checked(
        input,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn gated_mlp_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    params: GatedMlpI16Params<'_>,
    workspace: GatedMlpWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    validate_gated_mlp_shapes(input, params, &workspace, output)?;

    for token in 0..params.seq_len {
        let input_start = token.checked_mul(params.d_model)?;
        let input_end = input_start.checked_add(params.d_model)?;
        let hidden_start = token.checked_mul(params.hidden_dim)?;
        let hidden_end = hidden_start.checked_add(params.hidden_dim)?;
        let input_row = &input[input_start..input_end];

        linear_i16_i8_i16_per_channel_with_kernel_checked(
            input_row,
            params.up,
            &mut workspace.up[hidden_start..hidden_end],
            linear_kernel,
        )?;
        linear_i16_i8_i16_per_channel_with_kernel_checked(
            input_row,
            params.gate,
            &mut workspace.gate[hidden_start..hidden_end],
            linear_kernel,
        )?;

        for hidden_index in hidden_start..hidden_end {
            workspace.gated[hidden_index] =
                gated_activation_i16_q15(workspace.up[hidden_index], workspace.gate[hidden_index]);
        }

        linear_i16_i8_i16_per_channel_with_kernel_checked(
            &workspace.gated[hidden_start..hidden_end],
            params.down,
            &mut output[input_start..input_end],
            linear_kernel,
        )?;
    }

    Some(())
}

pub fn gated_mlp_backward_input_i16_q15_checked(
    grad_output: &[i16],
    params: GatedMlpI16Params<'_>,
    forward_up: &[i16],
    forward_gate: &[i16],
    scales: GatedMlpBackwardScales<'_>,
    workspace: GatedMlpBackwardWorkspace<'_>,
    grad_input: &mut [i16],
) -> Option<usize> {
    validate_gated_mlp_backward_shapes(
        grad_output,
        params,
        forward_up,
        forward_gate,
        scales,
        &workspace,
        grad_input,
    )?;

    let GatedMlpBackwardWorkspace {
        scaled_grad_output,
        grad_gated,
        grad_up,
        grad_gate,
        grad_up_input,
        grad_gate_input,
    } = workspace;

    let down_backward = LinearBackwardInputI16I8Params {
        weights: params.down.weights,
        forward_scales: params.down.scales,
        grad_input_scales: scales.down_to_hidden,
        input_dim: params.hidden_dim,
        output_dim: params.d_model,
    };
    let up_backward = LinearBackwardInputI16I8Params {
        weights: params.up.weights,
        forward_scales: params.up.scales,
        grad_input_scales: scales.up_to_input,
        input_dim: params.d_model,
        output_dim: params.hidden_dim,
    };
    let gate_backward = LinearBackwardInputI16I8Params {
        weights: params.gate.weights,
        forward_scales: params.gate.scales,
        grad_input_scales: scales.gate_to_input,
        input_dim: params.d_model,
        output_dim: params.hidden_dim,
    };

    let mut saturation_count = 0_usize;
    for token in 0..params.seq_len {
        let input_start = token.checked_mul(params.d_model)?;
        let input_end = input_start.checked_add(params.d_model)?;
        let hidden_start = token.checked_mul(params.hidden_dim)?;
        let hidden_end = hidden_start.checked_add(params.hidden_dim)?;

        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_output[input_start..input_end],
            down_backward,
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad_output[..params.d_model],
            },
            &mut grad_gated[hidden_start..hidden_end],
        )?;

        for hidden_index in hidden_start..hidden_end {
            let grad = gated_activation_backward_i16_q15(
                forward_up[hidden_index],
                forward_gate[hidden_index],
                grad_gated[hidden_index],
            );
            grad_up[hidden_index] = grad.up;
            grad_gate[hidden_index] = grad.gate;
        }

        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_up[hidden_start..hidden_end],
            up_backward,
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad_output[..params.hidden_dim],
            },
            &mut grad_up_input[input_start..input_end],
        )?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_gate[hidden_start..hidden_end],
            gate_backward,
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut scaled_grad_output[..params.hidden_dim],
            },
            &mut grad_gate_input[input_start..input_end],
        )?;

        saturation_count = saturation_count.checked_add(add_grad_rows_i16_checked(
            &grad_up_input[input_start..input_end],
            &grad_gate_input[input_start..input_end],
            &mut grad_input[input_start..input_end],
        )?)?;
    }

    Some(saturation_count)
}

pub fn gated_mlp_residual_block_i16_q15_checked(
    input: &[i16],
    params: GatedMlpI16Params<'_>,
    workspace: GatedMlpResidualWorkspace<'_>,
    output: &mut [i16],
) -> Option<usize> {
    gated_mlp_residual_block_i16_q15_with_linear_kernel_checked(
        input,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn gated_mlp_residual_block_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    params: GatedMlpI16Params<'_>,
    workspace: GatedMlpResidualWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<usize> {
    let GatedMlpResidualWorkspace { mlp, mlp_output } = workspace;

    if input.len() != output.len() || mlp_output.len() != input.len() {
        return None;
    }

    gated_mlp_i16_q15_with_linear_kernel_checked(input, params, mlp, mlp_output, linear_kernel)?;
    residual_add_i16_q15_checked(input, mlp_output, output)
}

pub fn prenorm_gated_mlp_residual_block_i16_q15_checked(
    input: &[i16],
    rms_weights_q15: &[i16],
    rms_eps: u64,
    params: GatedMlpI16Params<'_>,
    workspace: PreNormGatedMlpResidualWorkspace<'_>,
    output: &mut [i16],
) -> Option<usize> {
    prenorm_gated_mlp_residual_block_i16_q15_with_linear_kernel_checked(
        input,
        rms_weights_q15,
        rms_eps,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn prenorm_gated_mlp_residual_block_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    rms_weights_q15: &[i16],
    rms_eps: u64,
    params: GatedMlpI16Params<'_>,
    workspace: PreNormGatedMlpResidualWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<usize> {
    let total = params.seq_len.checked_mul(params.d_model)?;
    if input.len() != total
        || output.len() != total
        || workspace.normalized.len() != total
        || rms_weights_q15.len() != params.d_model
    {
        return None;
    }

    for token in 0..params.seq_len {
        let row_start = token.checked_mul(params.d_model)?;
        let row_end = row_start.checked_add(params.d_model)?;
        rms_norm_i16_q15_checked(
            &input[row_start..row_end],
            rms_weights_q15,
            rms_eps,
            &mut workspace.normalized[row_start..row_end],
        )?;
    }

    let GatedMlpResidualWorkspace { mlp, mlp_output } = workspace.residual;

    gated_mlp_i16_q15_with_linear_kernel_checked(
        workspace.normalized,
        params,
        mlp,
        mlp_output,
        linear_kernel,
    )?;
    residual_add_i16_q15_checked(input, mlp_output, output)
}

fn validate_gated_mlp_shapes(
    input: &[i16],
    params: GatedMlpI16Params<'_>,
    workspace: &GatedMlpWorkspace<'_>,
    output: &[i16],
) -> Option<()> {
    if !params.is_valid() {
        return None;
    }

    let total = params.seq_len.checked_mul(params.d_model)?;
    let hidden_total = params.seq_len.checked_mul(params.hidden_dim)?;
    if input.len() != total
        || output.len() != total
        || workspace.up.len() != hidden_total
        || workspace.gate.len() != hidden_total
        || workspace.gated.len() != hidden_total
    {
        return None;
    }

    Some(())
}

fn validate_gated_mlp_backward_shapes(
    grad_output: &[i16],
    params: GatedMlpI16Params<'_>,
    forward_up: &[i16],
    forward_gate: &[i16],
    scales: GatedMlpBackwardScales<'_>,
    workspace: &GatedMlpBackwardWorkspace<'_>,
    grad_input: &[i16],
) -> Option<()> {
    if !params.is_valid()
        || scales.down_to_hidden.len() != params.hidden_dim
        || scales.up_to_input.len() != params.d_model
        || scales.gate_to_input.len() != params.d_model
    {
        return None;
    }

    let total = params.seq_len.checked_mul(params.d_model)?;
    let hidden_total = params.seq_len.checked_mul(params.hidden_dim)?;
    let scaled_len = params.d_model.max(params.hidden_dim);
    if grad_output.len() != total
        || grad_input.len() != total
        || forward_up.len() != hidden_total
        || forward_gate.len() != hidden_total
        || workspace.scaled_grad_output.len() < scaled_len
        || workspace.grad_gated.len() != hidden_total
        || workspace.grad_up.len() != hidden_total
        || workspace.grad_gate.len() != hidden_total
        || workspace.grad_up_input.len() != total
        || workspace.grad_gate_input.len() != total
    {
        return None;
    }

    Some(())
}

fn add_grad_rows_i16_checked(left: &[i16], right: &[i16], output: &mut [i16]) -> Option<usize> {
    if left.len() != right.len() || left.len() != output.len() {
        return None;
    }

    let mut saturation_count = 0_usize;
    for ((&left, &right), out) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
        let wide = i32::from(left) + i32::from(right);
        if wide < i32::from(i16::MIN) || wide > i32::from(i16::MAX) {
            saturation_count = saturation_count.checked_add(1)?;
        }
        *out = saturating_add_i16(left, right);
    }

    Some(saturation_count)
}

fn residual_add_i16_q15_checked(skip: &[i16], block: &[i16], output: &mut [i16]) -> Option<usize> {
    if skip.len() != block.len() || skip.len() != output.len() {
        return None;
    }

    let mut saturation_count = 0_usize;
    for ((&skip, &block), out) in skip.iter().zip(block.iter()).zip(output.iter_mut()) {
        let wide = i32::from(skip) + i32::from(block);
        if wide < i32::from(i16::MIN) || wide > i32::from(i16::MAX) {
            saturation_count = saturation_count.checked_add(1)?;
        }

        *out = saturating_add_i16(skip, block);
    }

    Some(saturation_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedScale, RESIDUAL_Q15_SCALE};

    const IDENTITY_2: [i8; 4] = [1, 0, 0, 1];
    const UP_2_TO_4: [i8; 8] = [
        1, 0, //
        0, 1, //
        1, 1, //
        -1, 1,
    ];
    const DOWN_4_TO_2: [i8; 8] = [
        1, 0, 1, -1, //
        0, 1, 1, 1,
    ];
    const SCALES_2: [FixedScale; 2] = [RESIDUAL_Q15_SCALE; 2];
    const SCALES_4: [FixedScale; 4] = [RESIDUAL_Q15_SCALE; 4];

    fn params() -> GatedMlpI16Params<'static> {
        GatedMlpI16Params {
            up: LinearI16I8Params {
                weights: &UP_2_TO_4,
                bias: None,
                scales: &SCALES_4,
                input_dim: 2,
                output_dim: 4,
            },
            gate: LinearI16I8Params {
                weights: &UP_2_TO_4,
                bias: None,
                scales: &SCALES_4,
                input_dim: 2,
                output_dim: 4,
            },
            down: LinearI16I8Params {
                weights: &DOWN_4_TO_2,
                bias: None,
                scales: &SCALES_2,
                input_dim: 4,
                output_dim: 2,
            },
            seq_len: 1,
            d_model: 2,
            hidden_dim: 4,
        }
    }

    #[test]
    fn hard_silu_gate_is_bounded_and_monotonic() {
        assert_eq!(hard_silu_gate_q15(i16::MIN), 8192);
        assert_eq!(hard_silu_gate_q15(0), 16_384);
        assert_eq!(hard_silu_gate_q15(i16::MAX), 24_575);
        assert!(hard_silu_gate_q15(-1000) < hard_silu_gate_q15(1000));
    }

    #[test]
    fn hard_silu_preserves_sign_and_is_bounded() {
        assert!(hard_silu_q15(i16::MIN) < 0);
        assert_eq!(hard_silu_q15(0), 0);
        assert!(hard_silu_q15(i16::MAX) > 24_000);
    }

    #[test]
    fn hard_silu_derivative_tracks_middle_branch_in_i16_domain() {
        assert_eq!(hard_silu_derivative_q15(i16::MIN), 0);
        assert_eq!(hard_silu_derivative_q15(0), 16_384);
        assert_eq!(hard_silu_derivative_q15(i16::MAX), i16::MAX);
        assert_eq!(hard_silu_derivative_q15(-1), 16_383);
        assert_eq!(hard_silu_derivative_q15(1), 16_384);
        assert!(hard_silu_derivative_q15(-1000) < hard_silu_derivative_q15(1000));
    }

    #[test]
    fn gated_activation_uses_hard_silu_gate_value() {
        assert_eq!(gated_activation_i16_q15(10_000, 0), 0);
        assert!(gated_activation_i16_q15(10_000, 20_000) > 0);
        assert!(gated_activation_i16_q15(10_000, -20_000) < 0);
    }

    #[test]
    fn gated_activation_backward_applies_product_rule() {
        let grad = gated_activation_backward_i16_q15(16_384, 0, 16_384);

        assert_eq!(grad.up, 0);
        assert_eq!(grad.gate, 4096);

        let positive_gate = gated_activation_backward_i16_q15(16_384, 16_384, 16_384);
        assert!(positive_gate.up > 0);
        assert!(positive_gate.gate > grad.gate);
    }

    #[test]
    fn gated_mlp_runs_sequence_rows() {
        let input = [1000_i16, -500];
        let mut up = [0_i16; 4];
        let mut gate = [0_i16; 4];
        let mut gated = [0_i16; 4];
        let mut output = [0_i16; 2];

        assert!(
            gated_mlp_i16_q15_checked(
                &input,
                params(),
                GatedMlpWorkspace {
                    up: &mut up,
                    gate: &mut gate,
                    gated: &mut gated,
                },
                &mut output,
            )
            .is_some()
        );
        assert_ne!(output, [0, 0]);
    }

    #[test]
    fn gated_mlp_backward_chains_down_gate_and_input_transposes() {
        let input = [1000_i16, -500];
        let mut up = [0_i16; 4];
        let mut gate = [0_i16; 4];
        let mut gated = [0_i16; 4];
        let mut output = [0_i16; 2];

        gated_mlp_i16_q15_checked(
            &input,
            params(),
            GatedMlpWorkspace {
                up: &mut up,
                gate: &mut gate,
                gated: &mut gated,
            },
            &mut output,
        )
        .expect("forward");

        let grad_output = [100_i16, -50];
        let mut scaled_grad_output = [0_i32; 4];
        let mut grad_gated = [0_i16; 4];
        let mut grad_up = [0_i16; 4];
        let mut grad_gate = [0_i16; 4];
        let mut grad_up_input = [0_i16; 2];
        let mut grad_gate_input = [0_i16; 2];
        let mut grad_input = [0_i16; 2];

        let saturation_count = gated_mlp_backward_input_i16_q15_checked(
            &grad_output,
            params(),
            &up,
            &gate,
            GatedMlpBackwardScales {
                down_to_hidden: &SCALES_4,
                up_to_input: &SCALES_2,
                gate_to_input: &SCALES_2,
            },
            GatedMlpBackwardWorkspace {
                scaled_grad_output: &mut scaled_grad_output,
                grad_gated: &mut grad_gated,
                grad_up: &mut grad_up,
                grad_gate: &mut grad_gate,
                grad_up_input: &mut grad_up_input,
                grad_gate_input: &mut grad_gate_input,
            },
            &mut grad_input,
        );

        assert_eq!(saturation_count, Some(0));
        assert_eq!(grad_gated, [100, -50, 50, -150]);
        assert_ne!(grad_up, [0; 4]);
        assert_ne!(grad_gate, [0; 4]);
        assert_ne!(grad_input, [0; 2]);
    }

    #[test]
    fn gated_mlp_rejects_bad_shapes() {
        let input = [1000_i16, -500];
        let mut up = [0_i16; 3];
        let mut gate = [0_i16; 4];
        let mut gated = [0_i16; 4];
        let mut output = [0_i16; 2];

        assert_eq!(
            gated_mlp_i16_q15_checked(
                &input,
                params(),
                GatedMlpWorkspace {
                    up: &mut up,
                    gate: &mut gate,
                    gated: &mut gated,
                },
                &mut output,
            ),
            None
        );
    }

    #[test]
    fn residual_block_counts_saturation() {
        let params = GatedMlpI16Params {
            up: LinearI16I8Params {
                weights: &IDENTITY_2,
                bias: None,
                scales: &SCALES_2,
                input_dim: 2,
                output_dim: 2,
            },
            gate: LinearI16I8Params {
                weights: &IDENTITY_2,
                bias: None,
                scales: &SCALES_2,
                input_dim: 2,
                output_dim: 2,
            },
            down: LinearI16I8Params {
                weights: &IDENTITY_2,
                bias: None,
                scales: &SCALES_2,
                input_dim: 2,
                output_dim: 2,
            },
            seq_len: 1,
            d_model: 2,
            hidden_dim: 2,
        };
        let input = [30_000_i16, 30_000];
        let mut up = [0_i16; 2];
        let mut gate = [0_i16; 2];
        let mut gated = [0_i16; 2];
        let mut mlp_output = [0_i16; 2];
        let mut output = [0_i16; 2];

        assert_eq!(
            gated_mlp_residual_block_i16_q15_checked(
                &input,
                params,
                GatedMlpResidualWorkspace {
                    mlp: GatedMlpWorkspace {
                        up: &mut up,
                        gate: &mut gate,
                        gated: &mut gated,
                    },
                    mlp_output: &mut mlp_output,
                },
                &mut output,
            ),
            Some(2)
        );
        assert_eq!(output, [i16::MAX, i16::MAX]);
    }
}
