use crate::linear::{
    LinearI16I8Params, LinearKernel, linear_i16_i8_i16_per_channel_with_kernel_checked,
};
use crate::lut::{EXP2_NEG_FRAC_LUT_8BIT, normalize_u64_to_lut_index, recip_lut_8bit_q31};
use crate::numeric::{round_shift_rhu_i64, saturate_i16, saturating_add_i16};
use crate::rms_norm::rms_norm_i16_q15_checked;

pub const LOGIT_FRAC_BITS: u8 = 8;
pub const MASKED_LOGIT: i32 = i32::MIN;
pub const Q15_SHIFT: u8 = 15;

#[derive(Debug, Clone, Copy)]
pub struct SelfAttentionI16Params<'a> {
    pub q: LinearI16I8Params<'a>,
    pub k: LinearI16I8Params<'a>,
    pub v: LinearI16I8Params<'a>,
    pub o: LinearI16I8Params<'a>,
    pub seq_len: usize,
    pub d_model: usize,
    pub heads: usize,
    pub causal: bool,
}

pub struct SelfAttentionWorkspace<'a> {
    pub q: &'a mut [i16],
    pub k: &'a mut [i16],
    pub v: &'a mut [i16],
    pub context: &'a mut [i16],
    pub logits_q8: &'a mut [i32],
    pub probabilities_q15: &'a mut [i16],
}

pub struct AttentionResidualWorkspace<'a> {
    pub attention: SelfAttentionWorkspace<'a>,
    pub attention_output: &'a mut [i16],
}

pub struct PreNormAttentionResidualWorkspace<'a> {
    pub normalized: &'a mut [i16],
    pub residual: AttentionResidualWorkspace<'a>,
}

pub fn is_power_of_four(value: usize) -> bool {
    value != 0 && value.is_power_of_two() && value.trailing_zeros() % 2 == 0
}

pub fn sqrt_power_of_four_shift(value: usize) -> Option<u8> {
    if !is_power_of_four(value) {
        return None;
    }

    Some((value.trailing_zeros() / 2) as u8)
}

/// Returns true if `n` i16×i16 products (worst case 32768²) summed provably fit in i64.
///
/// Used for both QK dot products and probability×value accumulations.
/// Threshold: i64::MAX / 32768² ≈ 8.59 billion — no realistic model dimension reaches this.
#[inline]
fn fits_n_i16_products_in_i64(n: usize) -> bool {
    const MAX_PRODUCT: u128 = 32768 * 32768; // worst case: i16::MIN × i16::MIN
    (n as u128).saturating_mul(MAX_PRODUCT) <= i64::MAX as u128
}

pub fn attention_dot_q_k_i16_i32_checked(query: &[i16], key: &[i16]) -> Option<i32> {
    if query.len() != key.len() {
        return None;
    }

    let scale_shift = sqrt_power_of_four_shift(query.len())?;

    let acc = if fits_n_i16_products_in_i64(query.len()) {
        // Fast path: wrapping arithmetic so LLVM can auto-vectorize this loop.
        let mut acc = 0_i64;
        for (&q, &k) in query.iter().zip(key.iter()) {
            acc = acc.wrapping_add(i64::from(q) * i64::from(k));
        }
        acc
    } else {
        // Slow path: checked arithmetic for unusually large head dimensions.
        let mut acc = 0_i64;
        for (&q, &k) in query.iter().zip(key.iter()) {
            let product = i64::from(q) * i64::from(k);
            acc = acc.checked_add(product)?;
        }
        acc
    };

    i32::try_from(acc >> scale_shift).ok()
}

pub fn base2_exp_neg_q15(delta_logit_q8: i32) -> i16 {
    debug_assert!(delta_logit_q8 <= 0);

    if delta_logit_q8 >= 0 {
        return i16::MAX;
    }

    let magnitude = -(i64::from(delta_logit_q8));
    let integer_shift = magnitude >> LOGIT_FRAC_BITS;

    if integer_shift >= 15 {
        return 0;
    }

    let frac = (magnitude & ((1_i64 << LOGIT_FRAC_BITS) - 1)) as usize;
    EXP2_NEG_FRAC_LUT_8BIT[frac] >> integer_shift
}

pub fn reciprocal_sum_q31(sum: u64) -> Option<u32> {
    let normalized = normalize_u64_to_lut_index(sum)?;
    let base = u64::from(recip_lut_8bit_q31(normalized.mantissa));
    let reciprocal = if normalized.exponent >= 0 {
        let shift = u32::from(normalized.exponent as u16);
        if shift >= 63 { 0 } else { base >> shift }
    } else {
        let shift = u32::from((-normalized.exponent) as u16);
        base.checked_shl(shift).unwrap_or(u64::from(u32::MAX))
    };

    Some(reciprocal.min(u64::from(u32::MAX)) as u32)
}

pub fn base2_softmax_i32_q15(logits_q8: &[i32], output_q15: &mut [i16]) -> Option<u64> {
    if logits_q8.is_empty() || logits_q8.len() != output_q15.len() {
        return None;
    }

    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;

    let mut sum = 0_u64;
    for (&logit, out) in logits_q8.iter().zip(output_q15.iter_mut()) {
        if logit == MASKED_LOGIT {
            *out = 0;
            continue;
        }

        let delta = logit.saturating_sub(max_logit);
        let weight = base2_exp_neg_q15(delta);
        *out = weight;
        sum = sum.checked_add(weight as u64)?;
    }

    let inv_sum = i64::from(reciprocal_sum_q31(sum)?);

    for out in output_q15.iter_mut() {
        let product = i64::from(*out) * inv_sum;
        *out = saturate_i16(round_shift_rhu_i64(product, 16));
    }

    Some(sum)
}


pub fn attention_weight_v_i16_q15_checked(
    probabilities_q15: &[i16],
    values: &[i16],
    value_dim: usize,
    output: &mut [i16],
) -> Option<()> {
    if value_dim == 0
        || probabilities_q15.is_empty()
        || output.len() != value_dim
        || values.len() != probabilities_q15.len().checked_mul(value_dim)?
    {
        return None;
    }

    // Validate all probabilities are non-negative before the inner loop.
    if probabilities_q15.iter().any(|&p| p < 0) {
        return None;
    }

    let seq_len = probabilities_q15.len();

    if fits_n_i16_products_in_i64(seq_len) {
        // Fast path: wrapping arithmetic so LLVM can auto-vectorize the inner loop.
        for (out_index, out) in output.iter_mut().enumerate() {
            let mut acc = 0_i64;
            let mut row_offset = out_index;
            for &probability in probabilities_q15.iter() {
                let value = values[row_offset];
                acc = acc.wrapping_add(i64::from(probability) * i64::from(value));
                row_offset += value_dim;
            }
            *out = saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
        }
    } else {
        // Slow path: checked arithmetic for unusually large seq_len.
        for (out_index, out) in output.iter_mut().enumerate() {
            let mut acc = 0_i64;
            for (row, &probability) in probabilities_q15.iter().enumerate() {
                let value = values[row * value_dim + out_index];
                let product = i64::from(probability) * i64::from(value);
                acc = acc.checked_add(product)?;
            }
            *out = saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
        }
    }

    Some(())
}

pub fn attention_row_i16_q15_checked(
    query: &[i16],
    keys: &[i16],
    values: &[i16],
    key_dim: usize,
    value_dim: usize,
    mask: Option<&[bool]>,
    logits_q8: &mut [i32],
    probabilities_q15: &mut [i16],
    output: &mut [i16],
) -> Option<u64> {
    if key_dim == 0
        || value_dim == 0
        || query.len() != key_dim
        || keys.len() % key_dim != 0
        || values.len() % value_dim != 0
        || output.len() != value_dim
    {
        return None;
    }

    let key_count = keys.len() / key_dim;
    if key_count == 0
        || values.len() / value_dim != key_count
        || logits_q8.len() != key_count
        || probabilities_q15.len() != key_count
    {
        return None;
    }

    if let Some(mask) = mask
        && mask.len() != key_count
    {
        return None;
    }

    for key_index in 0..key_count {
        if mask.is_some_and(|mask| mask[key_index]) {
            logits_q8[key_index] = MASKED_LOGIT;
            continue;
        }

        let key_start = key_index.checked_mul(key_dim)?;
        let key_end = key_start.checked_add(key_dim)?;
        logits_q8[key_index] = attention_dot_q_k_i16_i32_checked(query, &keys[key_start..key_end])?;
    }

    let softmax_sum = base2_softmax_i32_q15(logits_q8, probabilities_q15)?;
    attention_weight_v_i16_q15_checked(probabilities_q15, values, value_dim, output)?;

    Some(softmax_sum)
}

pub fn self_attention_i16_q15_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: SelfAttentionWorkspace<'_>,
    output: &mut [i16],
) -> Option<()> {
    self_attention_i16_q15_with_linear_kernel_checked(
        input,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn self_attention_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: SelfAttentionWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    validate_self_attention_shapes(input, params, &workspace, output)?;

    let seq_len = params.seq_len;
    let d_model = params.d_model;
    let head_dim = d_model / params.heads;
    let total = seq_len.checked_mul(d_model)?;

    for token in 0..seq_len {
        let row_start = token.checked_mul(d_model)?;
        let row_end = row_start.checked_add(d_model)?;
        let input_row = &input[row_start..row_end];

        linear_i16_i8_i16_per_channel_with_kernel_checked(
            input_row,
            params.q,
            &mut workspace.q[row_start..row_end],
            linear_kernel,
        )?;
        linear_i16_i8_i16_per_channel_with_kernel_checked(
            input_row,
            params.k,
            &mut workspace.k[row_start..row_end],
            linear_kernel,
        )?;
        linear_i16_i8_i16_per_channel_with_kernel_checked(
            input_row,
            params.v,
            &mut workspace.v[row_start..row_end],
            linear_kernel,
        )?;
    }

    for token in 0..seq_len {
        for head in 0..params.heads {
            let head_offset = head.checked_mul(head_dim)?;
            let query_start = token.checked_mul(d_model)?.checked_add(head_offset)?;
            let query_end = query_start.checked_add(head_dim)?;
            let query = &workspace.q[query_start..query_end];

            for key_index in 0..seq_len {
                if params.causal && key_index > token {
                    workspace.logits_q8[key_index] = MASKED_LOGIT;
                    continue;
                }

                let key_start = key_index.checked_mul(d_model)?.checked_add(head_offset)?;
                let key_end = key_start.checked_add(head_dim)?;
                workspace.logits_q8[key_index] =
                    attention_dot_q_k_i16_i32_checked(query, &workspace.k[key_start..key_end])?;
            }

            base2_softmax_i32_q15(
                &workspace.logits_q8[..seq_len],
                &mut workspace.probabilities_q15[..seq_len],
            )?;

            // Pre-compute index bounds (checked once outside the hot loop).
            let base_v_offset = head_offset;
            let base_ctx_offset = token
                .checked_mul(d_model)?
                .checked_add(head_offset)?;

            // base2_softmax_i32_q15 guarantees non-negative outputs; no negative-prob scan needed.
            debug_assert!(workspace.probabilities_q15[..seq_len].iter().all(|&p| p >= 0));

            // Fast path: wrapping arithmetic safe because fits_n_i16_products_in_i64(seq_len).
            // seq_len is bounded by seq_len * d_model having been checked in validate_self_attention_shapes.
            debug_assert!(fits_n_i16_products_in_i64(seq_len));
            for out_index in 0..head_dim {
                let mut acc = 0_i64;

                for key_index in 0..seq_len {
                    let probability = workspace.probabilities_q15[key_index];
                    let value_index = key_index * d_model + base_v_offset + out_index;
                    acc = acc.wrapping_add(
                        i64::from(probability) * i64::from(workspace.v[value_index]),
                    );
                }

                workspace.context[base_ctx_offset + out_index] =
                    saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            }
        }
    }

    for token in 0..seq_len {
        let row_start = token.checked_mul(d_model)?;
        let row_end = row_start.checked_add(d_model)?;
        linear_i16_i8_i16_per_channel_with_kernel_checked(
            &workspace.context[row_start..row_end],
            params.o,
            &mut output[row_start..row_end],
            linear_kernel,
        )?;
    }

    debug_assert_eq!(total, output.len());
    Some(())
}

pub fn attention_residual_block_i16_q15_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: AttentionResidualWorkspace<'_>,
    output: &mut [i16],
) -> Option<usize> {
    attention_residual_block_i16_q15_with_linear_kernel_checked(
        input,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn attention_residual_block_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: AttentionResidualWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<usize> {
    let AttentionResidualWorkspace {
        attention,
        attention_output,
    } = workspace;

    if input.len() != output.len() || attention_output.len() != input.len() {
        return None;
    }

    self_attention_i16_q15_with_linear_kernel_checked(
        input,
        params,
        attention,
        attention_output,
        linear_kernel,
    )?;

    residual_add_i16_q15_checked(input, attention_output, output)
}

pub fn prenorm_attention_residual_block_i16_q15_checked(
    input: &[i16],
    rms_weights_q15: &[i16],
    rms_eps: u64,
    params: SelfAttentionI16Params<'_>,
    workspace: PreNormAttentionResidualWorkspace<'_>,
    output: &mut [i16],
) -> Option<usize> {
    prenorm_attention_residual_block_i16_q15_with_linear_kernel_checked(
        input,
        rms_weights_q15,
        rms_eps,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn prenorm_attention_residual_block_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    rms_weights_q15: &[i16],
    rms_eps: u64,
    params: SelfAttentionI16Params<'_>,
    workspace: PreNormAttentionResidualWorkspace<'_>,
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

    let AttentionResidualWorkspace {
        attention,
        attention_output,
    } = workspace.residual;

    self_attention_i16_q15_with_linear_kernel_checked(
        workspace.normalized,
        params,
        attention,
        attention_output,
        linear_kernel,
    )?;
    residual_add_i16_q15_checked(input, attention_output, output)
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

fn validate_self_attention_shapes(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: &SelfAttentionWorkspace<'_>,
    output: &[i16],
) -> Option<()> {
    if params.seq_len == 0 || params.d_model == 0 || params.heads == 0 {
        return None;
    }

    if params.d_model % params.heads != 0 {
        return None;
    }

    let head_dim = params.d_model / params.heads;
    if !is_power_of_four(head_dim) {
        return None;
    }

    if params.q.input_dim != params.d_model
        || params.k.input_dim != params.d_model
        || params.v.input_dim != params.d_model
        || params.o.input_dim != params.d_model
        || params.q.output_dim != params.d_model
        || params.k.output_dim != params.d_model
        || params.v.output_dim != params.d_model
        || params.o.output_dim != params.d_model
        || !params.q.is_valid()
        || !params.k.is_valid()
        || !params.v.is_valid()
        || !params.o.is_valid()
    {
        return None;
    }

    let total = params.seq_len.checked_mul(params.d_model)?;
    if input.len() != total
        || output.len() != total
        || workspace.q.len() != total
        || workspace.k.len() != total
        || workspace.v.len() != total
        || workspace.context.len() != total
        || workspace.logits_q8.len() < params.seq_len
        || workspace.probabilities_q15.len() < params.seq_len
    {
        return None;
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearI16I8Params, RESIDUAL_Q15_SCALE};
    use proptest::prelude::*;

    const IDENTITY_4: [i8; 16] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

    fn identity_params_4() -> LinearI16I8Params<'static> {
        static SCALES: [crate::FixedScale; 4] = [RESIDUAL_Q15_SCALE; 4];

        LinearI16I8Params {
            weights: &IDENTITY_4,
            bias: None,
            scales: &SCALES,
            input_dim: 4,
            output_dim: 4,
        }
    }

    #[test]
    fn validates_power_of_four_head_sizes() {
        assert!(is_power_of_four(1));
        assert!(is_power_of_four(4));
        assert!(is_power_of_four(16));
        assert!(is_power_of_four(64));
        assert!(!is_power_of_four(0));
        assert!(!is_power_of_four(2));
        assert!(!is_power_of_four(8));
    }

    #[test]
    fn qk_dot_uses_exact_sqrt_shift() {
        let query = [8_i16; 64];
        let key = [8_i16; 64];

        assert_eq!(attention_dot_q_k_i16_i32_checked(&query, &key), Some(512));
        assert_eq!(sqrt_power_of_four_shift(64), Some(3));
    }

    #[test]
    fn qk_dot_rejects_non_power_of_four_heads() {
        assert_eq!(
            attention_dot_q_k_i16_i32_checked(&[1_i16; 8], &[1_i16; 8]),
            None
        );
    }

    #[test]
    fn base2_exp_handles_integer_and_fractional_shifts() {
        assert_eq!(base2_exp_neg_q15(0), i16::MAX);
        assert_eq!(base2_exp_neg_q15(-(1 << LOGIT_FRAC_BITS)), 16383);
        assert!(base2_exp_neg_q15(-64) < i16::MAX);
        assert!(base2_exp_neg_q15(-64) > base2_exp_neg_q15(-128));
        assert_eq!(base2_exp_neg_q15(-(15 << LOGIT_FRAC_BITS)), 0);
    }

    #[test]
    fn reciprocal_sum_tracks_power_of_two_cases() {
        assert_eq!(reciprocal_sum_q31(1), Some(1_u32 << 31));
        assert_eq!(reciprocal_sum_q31(2), Some(1_u32 << 30));
        assert_eq!(reciprocal_sum_q31(128), Some(1_u32 << 24));
        assert_eq!(reciprocal_sum_q31(0), None);
    }

    #[test]
    fn base2_softmax_masks_by_annihilation() {
        let logits = [0_i32, MASKED_LOGIT, 0];
        let mut output = [0_i16; 3];

        assert!(base2_softmax_i32_q15(&logits, &mut output).is_some());
        assert!(output[0] > 16_000);
        assert_eq!(output[1], 0);
        assert!(output[2] > 16_000);
        assert!((i32::from(output[0]) - i32::from(output[2])).abs() <= 1);
    }

    #[test]
    fn base2_softmax_rejects_all_masked_rows() {
        let logits = [MASKED_LOGIT, MASKED_LOGIT];
        let mut output = [0_i16; 2];

        assert_eq!(base2_softmax_i32_q15(&logits, &mut output), None);
    }

    #[test]
    fn attention_weight_v_ignores_zero_probability_rows() {
        let probabilities = [i16::MAX, 0];
        let values = [100_i16, -100, 30_000, 30_000];
        let mut output = [0_i16; 2];

        assert!(
            attention_weight_v_i16_q15_checked(&probabilities, &values, 2, &mut output).is_some()
        );
        assert_eq!(output, [100, -100]);
    }

    #[test]
    fn attention_weight_v_rejects_bad_shapes_and_negative_probabilities() {
        let mut output = [0_i16; 2];

        assert_eq!(
            attention_weight_v_i16_q15_checked(&[i16::MAX], &[1_i16], 2, &mut output),
            None
        );
        assert_eq!(
            attention_weight_v_i16_q15_checked(&[-1_i16], &[1_i16, 2], 2, &mut output),
            None
        );
    }

    #[test]
    fn attention_row_applies_mask_before_softmax_and_value_weighting() {
        let query = [8_i16; 64];
        let mut keys = [0_i16; 128];
        keys[..64].fill(8);
        keys[64..].fill(8);

        let values = [100_i16, -100, 30_000, 30_000];
        let mask = [false, true];
        let mut logits = [0_i32; 2];
        let mut probabilities = [0_i16; 2];
        let mut output = [0_i16; 2];

        let sum = attention_row_i16_q15_checked(
            &query,
            &keys,
            &values,
            64,
            2,
            Some(&mask),
            &mut logits,
            &mut probabilities,
            &mut output,
        );

        assert!(sum.is_some());
        assert_eq!(logits[1], MASKED_LOGIT);
        assert_eq!(probabilities[1], 0);
        assert_eq!(output, [100, -100]);
    }

    #[test]
    fn attention_row_rejects_all_masked_rows() {
        let query = [1_i16; 64];
        let keys = [1_i16; 64];
        let values = [1_i16; 2];
        let mask = [true];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let mut output = [0_i16; 2];

        assert_eq!(
            attention_row_i16_q15_checked(
                &query,
                &keys,
                &values,
                64,
                2,
                Some(&mask),
                &mut logits,
                &mut probabilities,
                &mut output,
            ),
            None
        );
    }

    #[test]
    fn self_attention_identity_single_token_round_trips_through_projections() {
        let input = [100_i16, -200, 300, -400];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = SelfAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            logits_q8: &mut logits,
            probabilities_q15: &mut probabilities,
        };
        let mut output = [0_i16; 4];

        assert!(self_attention_i16_q15_checked(&input, params, workspace, &mut output).is_some());
        assert_eq!(output, input);
    }

    #[test]
    fn self_attention_causal_mask_blocks_future_value() {
        let input = [100_i16, 0, 0, 0, 0, 1000, 0, 0];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 2,
            d_model: 4,
            heads: 1,
            causal: true,
        };
        let mut q = [0_i16; 8];
        let mut k = [0_i16; 8];
        let mut v = [0_i16; 8];
        let mut context = [0_i16; 8];
        let mut logits = [0_i32; 2];
        let mut probabilities = [0_i16; 2];
        let workspace = SelfAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            logits_q8: &mut logits,
            probabilities_q15: &mut probabilities,
        };
        let mut output = [0_i16; 8];

        assert!(self_attention_i16_q15_checked(&input, params, workspace, &mut output).is_some());
        assert_eq!(&output[..4], &input[..4]);
    }

    #[test]
    fn self_attention_rejects_non_power_of_four_head_dim() {
        let weights = [1_i8; 4];
        let scales = [RESIDUAL_Q15_SCALE; 2];
        let linear = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 2,
            output_dim: 2,
        };
        let params = SelfAttentionI16Params {
            q: linear,
            k: linear,
            v: linear,
            o: linear,
            seq_len: 1,
            d_model: 2,
            heads: 1,
            causal: false,
        };
        let input = [1_i16, 2];
        let mut q = [0_i16; 2];
        let mut k = [0_i16; 2];
        let mut v = [0_i16; 2];
        let mut context = [0_i16; 2];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = SelfAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            logits_q8: &mut logits,
            probabilities_q15: &mut probabilities,
        };
        let mut output = [0_i16; 2];

        assert_eq!(
            self_attention_i16_q15_checked(&input, params, workspace, &mut output),
            None
        );
    }

    #[test]
    fn attention_residual_block_adds_attention_output_to_skip_path() {
        let input = [100_i16, -200, 300, -400];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut attention_output = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = AttentionResidualWorkspace {
            attention: SelfAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut context,
                logits_q8: &mut logits,
                probabilities_q15: &mut probabilities,
            },
            attention_output: &mut attention_output,
        };
        let mut output = [0_i16; 4];

        let saturation_count =
            attention_residual_block_i16_q15_checked(&input, params, workspace, &mut output);

        assert_eq!(saturation_count, Some(0));
        assert_eq!(output, [200, -400, 600, -800]);
    }

    #[test]
    fn attention_residual_block_counts_saturation() {
        let input = [20_000_i16, -20_000, 100, -100];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut attention_output = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = AttentionResidualWorkspace {
            attention: SelfAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut context,
                logits_q8: &mut logits,
                probabilities_q15: &mut probabilities,
            },
            attention_output: &mut attention_output,
        };
        let mut output = [0_i16; 4];

        let saturation_count =
            attention_residual_block_i16_q15_checked(&input, params, workspace, &mut output);

        assert_eq!(saturation_count, Some(2));
        assert_eq!(output, [i16::MAX, i16::MIN, 200, -200]);
    }

    #[test]
    fn attention_residual_block_rejects_bad_output_shape() {
        let input = [1_i16, 2, 3, 4];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut attention_output = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = AttentionResidualWorkspace {
            attention: SelfAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut context,
                logits_q8: &mut logits,
                probabilities_q15: &mut probabilities,
            },
            attention_output: &mut attention_output,
        };
        let mut output = [0_i16; 3];

        assert_eq!(
            attention_residual_block_i16_q15_checked(&input, params, workspace, &mut output),
            None
        );
    }

    #[test]
    fn prenorm_attention_residual_block_normalizes_attention_input_only() {
        let input = [100_i16, 100, 100, 100];
        let rms_weights = [1000_i16; 4];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut normalized = [0_i16; 4];
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut attention_output = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = PreNormAttentionResidualWorkspace {
            normalized: &mut normalized,
            residual: AttentionResidualWorkspace {
                attention: SelfAttentionWorkspace {
                    q: &mut q,
                    k: &mut k,
                    v: &mut v,
                    context: &mut context,
                    logits_q8: &mut logits,
                    probabilities_q15: &mut probabilities,
                },
                attention_output: &mut attention_output,
            },
        };
        let mut output = [0_i16; 4];

        let saturation_count = prenorm_attention_residual_block_i16_q15_checked(
            &input,
            &rms_weights,
            0,
            params,
            workspace,
            &mut output,
        );

        assert_eq!(saturation_count, Some(0));
        assert!(
            normalized
                .iter()
                .all(|&value| (990..=1010).contains(&value))
        );
        assert!(output.iter().all(|&value| (1090..=1110).contains(&value)));
    }

    #[test]
    fn prenorm_attention_residual_block_rejects_bad_rms_weight_shape() {
        let input = [100_i16, 100, 100, 100];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: false,
        };
        let mut normalized = [0_i16; 4];
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut attention_output = [0_i16; 4];
        let mut logits = [0_i32; 1];
        let mut probabilities = [0_i16; 1];
        let workspace = PreNormAttentionResidualWorkspace {
            normalized: &mut normalized,
            residual: AttentionResidualWorkspace {
                attention: SelfAttentionWorkspace {
                    q: &mut q,
                    k: &mut k,
                    v: &mut v,
                    context: &mut context,
                    logits_q8: &mut logits,
                    probabilities_q15: &mut probabilities,
                },
                attention_output: &mut attention_output,
            },
        };
        let mut output = [0_i16; 4];

        assert_eq!(
            prenorm_attention_residual_block_i16_q15_checked(
                &input,
                &[1000_i16; 3],
                0,
                params,
                workspace,
                &mut output,
            ),
            None
        );
    }

    proptest! {
        #[test]
        fn base2_softmax_never_panics_for_small_rows(
            logits in proptest::collection::vec(-4096_i32..=4096, 1..64),
        ) {
            let mut output = [0_i16; 64];
            let len = logits.len();
            let sum = base2_softmax_i32_q15(&logits, &mut output[..len]);

            prop_assert!(sum.is_some());
            prop_assert!(output[..len].iter().all(|&value| value >= 0));
        }
    }
}
