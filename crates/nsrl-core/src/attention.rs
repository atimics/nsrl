use crate::linear::{
    LinearI16I8Params, LinearKernel, linear_i16_i8_i16_per_channel_with_kernel_checked,
};
use crate::lut::{EXP2_NEG_FRAC_LUT_8BIT, normalize_u64_to_lut_index, recip_lut_8bit_q31};
use crate::numeric::{
    FixedScale, MAX_RIGHT_SHIFT, RESIDUAL_Q15_SCALE, residual_add_i16_q15_checked,
    round_shift_rhu_i64, saturate_i16,
};
use crate::rms_norm::rms_norm_i16_q15_checked;

pub const LOGIT_FRAC_BITS: u8 = 8;
pub const MASKED_LOGIT: i32 = i32::MIN;
pub const Q15_SHIFT: u8 = 15;
pub const DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS: u64 = 32_000;
const NLL_LOG2_FRACTIONAL_BITS: u32 = 20;

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

pub struct LinearAttentionWorkspace<'a> {
    pub q: &'a mut [i16],
    pub k: &'a mut [i16],
    pub v: &'a mut [i16],
    pub context: &'a mut [i16],
    pub state_kv: &'a mut [i64],
    pub key_sums: &'a mut [i64],
}

pub struct LinearAttentionStepWorkspace<'a> {
    pub q: &'a mut [i16],
    pub k: &'a mut [i16],
    pub v: &'a mut [i16],
    pub context: &'a mut [i16],
}

pub struct LinearAttentionTttStepWorkspace<'a> {
    pub q: &'a mut [i16],
    pub k: &'a mut [i16],
    pub v: &'a mut [i16],
    pub context: &'a mut [i16],
    pub prediction: &'a mut [i16],
}

pub struct LinearAttentionState<'a> {
    pub state_kv: &'a mut [i64],
    pub key_sums: &'a mut [i64],
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
    value != 0 && value.is_power_of_two() && value.trailing_zeros().is_multiple_of(2)
}

pub fn sqrt_power_of_four_shift(value: usize) -> Option<u8> {
    if !is_power_of_four(value) {
        return None;
    }

    Some((value.trailing_zeros() / 2) as u8)
}

/// Returns true if `n` i16-by-i16 products summed provably fit in i64.
///
/// Used for both QK dot products and probability-by-value accumulations.
/// The worst case is 32768 * 32768, so the threshold is
/// i64::MAX / 1_073_741_824 = 8_589_934_591 products.
#[inline]
fn fits_n_i16_products_in_i64(n: usize) -> bool {
    const MAX_PRODUCT: u128 = 32768 * 32768;
    (n as u128).saturating_mul(MAX_PRODUCT) <= i64::MAX as u128
}

pub fn attention_dot_q_k_i16_i32_checked(query: &[i16], key: &[i16]) -> Option<i32> {
    if query.len() != key.len() {
        return None;
    }

    let scale_shift = sqrt_power_of_four_shift(query.len())?;
    attention_dot_q_k_i16_i32_with_shift(query, key, scale_shift)
}

/// QK dot product with a caller-supplied `scale_shift`.
///
/// `scale_shift` is `sqrt_power_of_four_shift(head_dim)` and is loop-invariant
/// across every key in an attention row, so callers hoist it out of the key loop
/// rather than recomputing `trailing_zeros` per dot product.
///
/// Callers must pass `query.len() == key.len()`; the zip stops at the shorter slice.
#[inline]
fn attention_dot_q_k_i16_i32_with_shift(
    query: &[i16],
    key: &[i16],
    scale_shift: u8,
) -> Option<i32> {
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

/// Q47 counterpart of [`base2_exp_neg_q15`] for objective evaluation and
/// fixed-mass proposal construction. This preserves the committed Q15
/// fractional table while retaining another 32 exponent bits, so logit gaps
/// below 47 bits remain observable without changing the deployed softmax.
pub fn base2_exp_neg_q47(delta_logit_q8: i32) -> u64 {
    debug_assert!(delta_logit_q8 <= 0);

    if delta_logit_q8 >= 0 {
        return (i16::MAX as u64) << 32;
    }

    let magnitude = -(i64::from(delta_logit_q8));
    let integer_shift = magnitude >> LOGIT_FRAC_BITS;
    if integer_shift >= 47 {
        return 0;
    }

    let frac = (magnitude & ((1_i64 << LOGIT_FRAC_BITS) - 1)) as usize;
    (u64::from(EXP2_NEG_FRAC_LUT_8BIT[frac] as u16) << 32) >> integer_shift
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

/// Reciprocal implementation used when retaining normalized probabilities in
/// Q31. The legacy variant reproduces the frozen Q31 path exactly. The Q47
/// variants retain sixteen more reciprocal bits before the probability product
/// is requantized to Q31.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxNormalization {
    LegacyQ31Lut,
    Q47Lut,
    Q47Newton1,
    Q47Exact,
}

impl SoftmaxNormalization {
    /// Stable artifact/CLI identifier for the normalization contract.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyQ31Lut => "legacy_q31_lut",
            Self::Q47Lut => "q47_lut",
            Self::Q47Newton1 => "q47_newton1",
            Self::Q47Exact => "q47_exact_division",
        }
    }
}

/// Return the established 8-bit-LUT reciprocal in Q47 without first discarding
/// the low sixteen bits needed by a Q31 probability product.
pub fn reciprocal_sum_q47_lut(sum: u64) -> Option<u64> {
    let normalized = normalize_u64_to_lut_index(sum)?;
    let base = u64::from(recip_lut_8bit_q31(normalized.mantissa)).checked_shl(16)?;
    if normalized.exponent >= 0 {
        let shift = u32::from(normalized.exponent as u16);
        Some(if shift >= 64 { 0 } else { base >> shift })
    } else {
        let shift = u32::from((-normalized.exponent) as u16);
        base.checked_shl(shift)
    }
}

/// Apply one integer Newton-Raphson refinement to the Q47 LUT reciprocal.
///
/// For `r ~= 1 / sum`, the update is `r * (2 - sum * r)`. Every intermediate
/// is represented in Q47 and evaluated in `u128` so the result is deterministic
/// and does not depend on floating-point support.
pub fn reciprocal_sum_q47_newton1(sum: u64) -> Option<u64> {
    let reciprocal = u128::from(reciprocal_sum_q47_lut(sum)?);
    let scale = 1_u128 << 47;
    let sum_times_reciprocal = u128::from(sum).checked_mul(reciprocal)?;
    let correction = scale.checked_mul(2)?.checked_sub(sum_times_reciprocal)?;
    let refined = reciprocal
        .checked_mul(correction)?
        .checked_add(1_u128 << 46)?
        >> 47;
    u64::try_from(refined).ok()
}

/// Return the rounded exact integer reciprocal in Q47. This is an audit ceiling
/// rather than the default implementation: it establishes how much error is
/// caused by reciprocal approximation versus per-probability rounding.
pub fn reciprocal_sum_q47_exact(sum: u64) -> Option<u64> {
    if sum == 0 {
        return None;
    }
    let numerator = (1_u128 << 47).checked_add(u128::from(sum >> 1))?;
    u64::try_from(numerator / u128::from(sum)).ok()
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

/// Evaluate the same integer base-2 softmax as [`base2_softmax_i32_q15`], but
/// retain its normalized probability product in Q31. This is primarily useful
/// for resolution audits and wider-gradient experiments; attention kernels
/// continue to consume the established Q15 representation.
pub fn base2_softmax_i32_q31(logits_q8: &[i32], output_q31: &mut [u32]) -> Option<u64> {
    base2_softmax_i32_q31_with_normalization(
        logits_q8,
        output_q31,
        SoftmaxNormalization::LegacyQ31Lut,
    )
}

/// Evaluate integer base-2 softmax in Q31 with an explicitly selected
/// normalization implementation.
pub fn base2_softmax_i32_q31_with_normalization(
    logits_q8: &[i32],
    output_q31: &mut [u32],
    normalization: SoftmaxNormalization,
) -> Option<u64> {
    if logits_q8.is_empty() || logits_q8.len() != output_q31.len() {
        return None;
    }

    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;

    let mut sum = 0_u64;
    for (&logit, out) in logits_q8.iter().zip(output_q31.iter_mut()) {
        if logit == MASKED_LOGIT {
            *out = 0;
            continue;
        }

        let delta = logit.saturating_sub(max_logit);
        let weight = base2_exp_neg_q15(delta);
        *out = u32::from(weight as u16);
        sum = sum.checked_add(u64::from(weight as u16))?;
    }

    match normalization {
        SoftmaxNormalization::LegacyQ31Lut => {
            let inv_sum = u64::from(reciprocal_sum_q31(sum)?);
            for out in output_q31.iter_mut() {
                let product = u64::from(*out).checked_mul(inv_sum)?;
                *out = product.min(i32::MAX as u64) as u32;
            }
        }
        SoftmaxNormalization::Q47Lut
        | SoftmaxNormalization::Q47Newton1
        | SoftmaxNormalization::Q47Exact => {
            let inv_sum = match normalization {
                SoftmaxNormalization::Q47Lut => reciprocal_sum_q47_lut(sum)?,
                SoftmaxNormalization::Q47Newton1 => reciprocal_sum_q47_newton1(sum)?,
                SoftmaxNormalization::Q47Exact => reciprocal_sum_q47_exact(sum)?,
                SoftmaxNormalization::LegacyQ31Lut => unreachable!(),
            };
            for out in output_q31.iter_mut() {
                let product = u128::from(*out).checked_mul(u128::from(inv_sum))?;
                let rounded = product.checked_add(1_u128 << 15)? >> 16;
                *out = rounded.min(i32::MAX as u128) as u32;
            }
        }
    }

    Some(sum)
}

/// Evaluate the negative log-likelihood of `target` under the same base-2
/// exponent approximation used by the integer softmax, without first rounding
/// the normalized target probability to Q15 or Q31.
///
/// The declared integer objective is
/// `log2(sum_i weight_i) - log2(weight_target)`, rounded to millibits. A target
/// whose exponent approximation annihilates to zero receives the caller-bound
/// zero-probability floor. Because the reciprocal is not part of this
/// calculation, the result is shift-invariant and independent of probability
/// normalization error.
pub fn base2_softmax_nll_millibits(
    logits_q8: &[i32],
    target: usize,
    zero_probability_floor_millibits: u64,
) -> Option<u64> {
    let zero_floor_q20 = zero_probability_floor_millibits
        .checked_mul(1_u64 << NLL_LOG2_FRACTIONAL_BITS)?
        .checked_add(500)?
        / 1_000;
    let loss_q20 = base2_softmax_nll_q20(logits_q8, target, zero_floor_q20)?;
    loss_q20
        .checked_mul(1_000)?
        .checked_add(1_u64 << (NLL_LOG2_FRACTIONAL_BITS - 1))
        .map(|rounded| rounded >> NLL_LOG2_FRACTIONAL_BITS)
}

/// Q20-bit counterpart of [`base2_softmax_nll_millibits`] for exact lattice
/// comparisons that would be hidden by millibit reporting resolution.
pub fn base2_softmax_nll_q20(
    logits_q8: &[i32],
    target: usize,
    zero_probability_floor_q20: u64,
) -> Option<u64> {
    if logits_q8.is_empty() || target >= logits_q8.len() {
        return None;
    }
    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;
    if logits_q8[target] == MASKED_LOGIT {
        return Some(zero_probability_floor_q20);
    }

    let mut weight_sum = 0_u64;
    let mut target_weight = 0_u64;
    for (index, &logit) in logits_q8.iter().enumerate() {
        if logit == MASKED_LOGIT {
            continue;
        }
        let weight = u64::from(base2_exp_neg_q15(logit.saturating_sub(max_logit)) as u16);
        weight_sum = weight_sum.checked_add(weight)?;
        if index == target {
            target_weight = weight;
        }
    }
    if target_weight == 0 {
        return Some(zero_probability_floor_q20);
    }

    let denominator_log2_q20 = log2_u64_q20(weight_sum)?;
    let numerator_log2_q20 = log2_u64_q20(target_weight)?;
    denominator_log2_q20.checked_sub(numerator_log2_q20)
}

/// Wide Q47 logit-anchored NLL from MJ-05. This is a separately versioned
/// observation objective; it does not replace the deployed Q15 softmax path.
pub fn base2_softmax_nll_q47_q20(
    logits_q8: &[i32],
    target: usize,
    zero_probability_floor_q20: u64,
) -> Option<u64> {
    if logits_q8.is_empty() || target >= logits_q8.len() {
        return None;
    }
    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;
    if logits_q8[target] == MASKED_LOGIT {
        return Some(zero_probability_floor_q20);
    }

    let mut weight_sum = 0_u64;
    let mut target_weight = 0_u64;
    for (index, &logit) in logits_q8.iter().enumerate() {
        if logit == MASKED_LOGIT {
            continue;
        }
        let weight = base2_exp_neg_q47(logit.saturating_sub(max_logit));
        weight_sum = weight_sum.checked_add(weight)?;
        if index == target {
            target_weight = weight;
        }
    }
    if target_weight == 0 {
        return Some(zero_probability_floor_q20);
    }

    log2_u64_q20(weight_sum)?.checked_sub(log2_u64_q20(target_weight)?)
}

/// Q32 observation counterpart of [`base2_softmax_nll_q47_q20`]. The deployed
/// logits and Q47 exponential weights are identical; only the final logarithm
/// retains another twelve fractional bits. This is an audit objective, not a
/// training-objective or forward-path change.
pub fn base2_softmax_nll_q47_q32(
    logits_q8: &[i32],
    target: usize,
    zero_probability_floor_q32: u64,
) -> Option<u64> {
    if logits_q8.is_empty() || target >= logits_q8.len() {
        return None;
    }
    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;
    if logits_q8[target] == MASKED_LOGIT {
        return Some(zero_probability_floor_q32);
    }

    let mut weight_sum = 0_u64;
    let mut target_weight = 0_u64;
    for (index, &logit) in logits_q8.iter().enumerate() {
        if logit == MASKED_LOGIT {
            continue;
        }
        let weight = base2_exp_neg_q47(logit.saturating_sub(max_logit));
        weight_sum = weight_sum.checked_add(weight)?;
        if index == target {
            target_weight = weight;
        }
    }
    if target_weight == 0 {
        return Some(zero_probability_floor_q32);
    }

    log2_u64_fixed(weight_sum, 32)?.checked_sub(log2_u64_fixed(target_weight, 32)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Base2SoftmaxNllQ47Components {
    pub weight_sum: u64,
    pub target_weight: u64,
    pub denominator_log2_q20: u64,
    pub target_log2_q20: u64,
    pub denominator_log2_q32: u64,
    pub target_log2_q32: u64,
}

/// Returns the exact Q47-weight denominator/target logarithm components at the
/// Q20 and Q32 observation grids. The Q20 components are required to equal the
/// Q32 prefixes, making boundary crossings and cancellation auditable without
/// changing the softmax used by inference.
pub fn base2_softmax_nll_q47_components(
    logits_q8: &[i32],
    target: usize,
) -> Option<Base2SoftmaxNllQ47Components> {
    if logits_q8.is_empty() || target >= logits_q8.len() {
        return None;
    }
    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;
    if logits_q8[target] == MASKED_LOGIT {
        return None;
    }
    let mut weight_sum = 0_u64;
    let mut target_weight = 0_u64;
    for (index, &logit) in logits_q8.iter().enumerate() {
        if logit == MASKED_LOGIT {
            continue;
        }
        let weight = base2_exp_neg_q47(logit.saturating_sub(max_logit));
        weight_sum = weight_sum.checked_add(weight)?;
        if index == target {
            target_weight = weight;
        }
    }
    if target_weight == 0 {
        return None;
    }
    let denominator_log2_q20 = log2_u64_fixed(weight_sum, 20)?;
    let target_log2_q20 = log2_u64_fixed(target_weight, 20)?;
    let denominator_log2_q32 = log2_u64_fixed(weight_sum, 32)?;
    let target_log2_q32 = log2_u64_fixed(target_weight, 32)?;
    if denominator_log2_q32 >> 12 != denominator_log2_q20
        || target_log2_q32 >> 12 != target_log2_q20
    {
        return None;
    }
    Some(Base2SoftmaxNllQ47Components {
        weight_sum,
        target_weight,
        denominator_log2_q20,
        target_log2_q20,
        denominator_log2_q32,
        target_log2_q32,
    })
}

fn log2_u64_q20(value: u64) -> Option<u64> {
    log2_u64_fixed(value, NLL_LOG2_FRACTIONAL_BITS)
}

fn log2_u64_fixed(value: u64, fractional_bits: u32) -> Option<u64> {
    if value == 0 {
        return None;
    }
    if fractional_bits > 40 {
        return None;
    }
    let integer_log2 = u64::BITS - 1 - value.leading_zeros();
    let mut normalized_q63 = u128::from(value) << (63 - integer_log2);
    let mut fractional = 0_u64;
    for bit in (0..fractional_bits).rev() {
        normalized_q63 = normalized_q63.checked_mul(normalized_q63)? >> 63;
        if normalized_q63 >= (1_u128 << 64) {
            normalized_q63 >>= 1;
            fractional |= 1_u64 << bit;
        }
    }
    Some((u64::from(integer_log2) << fractional_bits) | fractional)
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
        // Tile across output channels so each probability pass reads contiguous value lanes.
        let mut out_index = 0_usize;
        while out_index + 4 <= value_dim {
            let mut acc0 = 0_i64;
            let mut acc1 = 0_i64;
            let mut acc2 = 0_i64;
            let mut acc3 = 0_i64;
            let mut row_offset = out_index;

            for &probability in probabilities_q15.iter() {
                let probability = i64::from(probability);
                acc0 = acc0.wrapping_add(probability * i64::from(values[row_offset]));
                acc1 = acc1.wrapping_add(probability * i64::from(values[row_offset + 1]));
                acc2 = acc2.wrapping_add(probability * i64::from(values[row_offset + 2]));
                acc3 = acc3.wrapping_add(probability * i64::from(values[row_offset + 3]));
                row_offset += value_dim;
            }

            output[out_index] = saturate_i16(round_shift_rhu_i64(acc0, Q15_SHIFT));
            output[out_index + 1] = saturate_i16(round_shift_rhu_i64(acc1, Q15_SHIFT));
            output[out_index + 2] = saturate_i16(round_shift_rhu_i64(acc2, Q15_SHIFT));
            output[out_index + 3] = saturate_i16(round_shift_rhu_i64(acc3, Q15_SHIFT));
            out_index += 4;
        }

        while out_index < value_dim {
            let mut acc = 0_i64;
            let mut row_offset = out_index;
            for &probability in probabilities_q15.iter() {
                let value = values[row_offset];
                acc = acc.wrapping_add(i64::from(probability) * i64::from(value));
                row_offset += value_dim;
            }
            output[out_index] = saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            out_index += 1;
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

#[allow(clippy::too_many_arguments)]
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
        || !keys.len().is_multiple_of(key_dim)
        || !values.len().is_multiple_of(value_dim)
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

    // Loop-invariant across every key; hoisted out of the per-key dot product.
    let scale_shift = sqrt_power_of_four_shift(key_dim)?;
    for key_index in 0..key_count {
        if mask.is_some_and(|mask| mask[key_index]) {
            logits_q8[key_index] = MASKED_LOGIT;
            continue;
        }

        let key_start = key_index.checked_mul(key_dim)?;
        let key_end = key_start.checked_add(key_dim)?;
        logits_q8[key_index] =
            attention_dot_q_k_i16_i32_with_shift(query, &keys[key_start..key_end], scale_shift)?;
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
    // QK scaling shift depends only on head_dim; compute once for the whole call.
    let scale_shift = sqrt_power_of_four_shift(head_dim)?;

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

            let effective_len = if params.causal { token + 1 } else { seq_len };
            for key_index in 0..effective_len {
                let key_start = key_index.checked_mul(d_model)?.checked_add(head_offset)?;
                let key_end = key_start.checked_add(head_dim)?;
                workspace.logits_q8[key_index] = attention_dot_q_k_i16_i32_with_shift(
                    query,
                    &workspace.k[key_start..key_end],
                    scale_shift,
                )?;
            }

            base2_softmax_i32_q15(
                &workspace.logits_q8[..effective_len],
                &mut workspace.probabilities_q15[..effective_len],
            )?;

            // Pre-compute index bounds (checked once outside the hot loop).
            let base_v_offset = head_offset;
            let base_ctx_offset = token.checked_mul(d_model)?.checked_add(head_offset)?;

            // base2_softmax_i32_q15 guarantees non-negative outputs; no negative-prob scan needed.
            debug_assert!(
                workspace.probabilities_q15[..effective_len]
                    .iter()
                    .all(|&p| p >= 0)
            );

            // Fast path: wrapping arithmetic safe because fits_n_i16_products_in_i64(seq_len).
            // seq_len is bounded by seq_len * d_model having been checked in validate_self_attention_shapes.
            debug_assert!(fits_n_i16_products_in_i64(seq_len));
            let mut out_index = 0_usize;
            while out_index + 4 <= head_dim {
                let mut acc0 = 0_i64;
                let mut acc1 = 0_i64;
                let mut acc2 = 0_i64;
                let mut acc3 = 0_i64;

                for key_index in 0..effective_len {
                    let probability = i64::from(workspace.probabilities_q15[key_index]);
                    let value_index = key_index * d_model + base_v_offset + out_index;
                    acc0 = acc0.wrapping_add(probability * i64::from(workspace.v[value_index]));
                    acc1 = acc1.wrapping_add(probability * i64::from(workspace.v[value_index + 1]));
                    acc2 = acc2.wrapping_add(probability * i64::from(workspace.v[value_index + 2]));
                    acc3 = acc3.wrapping_add(probability * i64::from(workspace.v[value_index + 3]));
                }

                workspace.context[base_ctx_offset + out_index] =
                    saturate_i16(round_shift_rhu_i64(acc0, Q15_SHIFT));
                workspace.context[base_ctx_offset + out_index + 1] =
                    saturate_i16(round_shift_rhu_i64(acc1, Q15_SHIFT));
                workspace.context[base_ctx_offset + out_index + 2] =
                    saturate_i16(round_shift_rhu_i64(acc2, Q15_SHIFT));
                workspace.context[base_ctx_offset + out_index + 3] =
                    saturate_i16(round_shift_rhu_i64(acc3, Q15_SHIFT));
                out_index += 4;
            }

            while out_index < head_dim {
                let mut acc = 0_i64;

                for key_index in 0..effective_len {
                    let probability = workspace.probabilities_q15[key_index];
                    let value_index = key_index * d_model + base_v_offset + out_index;
                    acc = acc
                        .wrapping_add(i64::from(probability) * i64::from(workspace.v[value_index]));
                }

                workspace.context[base_ctx_offset + out_index] =
                    saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
                out_index += 1;
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

pub fn linear_attention_i16_q15_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionWorkspace<'_>,
    output: &mut [i16],
) -> Option<()> {
    linear_attention_i16_q15_with_linear_kernel_checked(
        input,
        params,
        workspace,
        output,
        LinearKernel::GenericI8,
    )
}

pub fn linear_attention_i16_q15_with_linear_kernel_checked(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    linear_attention_i16_q15_decay_kernel_impl(
        input,
        params,
        RESIDUAL_Q15_SCALE,
        workspace,
        output,
        linear_kernel,
    )
}

fn linear_attention_i16_q15_decay_kernel_impl(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    decay: FixedScale,
    workspace: LinearAttentionWorkspace<'_>,
    output: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    validate_linear_attention_shapes(input, params, &workspace, output)?;
    if decay.multiplier < 0 || decay.right_shift > MAX_RIGHT_SHIFT {
        return None;
    }

    let seq_len = params.seq_len;
    let d_model = params.d_model;
    let head_dim = d_model / params.heads;
    let state_len = params.heads.checked_mul(head_dim.checked_mul(head_dim)?)?;
    let key_sum_len = params.heads.checked_mul(head_dim)?;

    workspace.state_kv[..state_len].fill(0);
    workspace.key_sums[..key_sum_len].fill(0);

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

    if params.causal {
        for token in 0..seq_len {
            for head in 0..params.heads {
                let head_offset = head.checked_mul(head_dim)?;
                let row_base = token.checked_mul(d_model)?.checked_add(head_offset)?;
                let head_state_start = head.checked_mul(head_dim.checked_mul(head_dim)?)?;
                let head_state_end =
                    head_state_start.checked_add(head_dim.checked_mul(head_dim)?)?;
                let head_sum_start = head.checked_mul(head_dim)?;
                let head_sum_end = head_sum_start.checked_add(head_dim)?;

                accumulate_linear_attention_state_i16_checked(
                    &workspace.k[row_base..row_base + head_dim],
                    &workspace.v[row_base..row_base + head_dim],
                    &mut workspace.state_kv[head_state_start..head_state_end],
                    &mut workspace.key_sums[head_sum_start..head_sum_end],
                    head_dim,
                    decay,
                )?;
                project_linear_attention_state_i16_checked(
                    &workspace.q[row_base..row_base + head_dim],
                    &workspace.state_kv[head_state_start..head_state_end],
                    &workspace.key_sums[head_sum_start..head_sum_end],
                    head_dim,
                    &mut workspace.context[row_base..row_base + head_dim],
                )?;
            }
        }
    } else {
        for token in 0..seq_len {
            for head in 0..params.heads {
                let head_offset = head.checked_mul(head_dim)?;
                let row_base = token.checked_mul(d_model)?.checked_add(head_offset)?;
                let head_state_start = head.checked_mul(head_dim.checked_mul(head_dim)?)?;
                let head_state_end =
                    head_state_start.checked_add(head_dim.checked_mul(head_dim)?)?;
                let head_sum_start = head.checked_mul(head_dim)?;
                let head_sum_end = head_sum_start.checked_add(head_dim)?;

                accumulate_linear_attention_state_i16_checked(
                    &workspace.k[row_base..row_base + head_dim],
                    &workspace.v[row_base..row_base + head_dim],
                    &mut workspace.state_kv[head_state_start..head_state_end],
                    &mut workspace.key_sums[head_sum_start..head_sum_end],
                    head_dim,
                    decay,
                )?;
            }
        }

        for token in 0..seq_len {
            for head in 0..params.heads {
                let head_offset = head.checked_mul(head_dim)?;
                let row_base = token.checked_mul(d_model)?.checked_add(head_offset)?;
                let head_state_start = head.checked_mul(head_dim.checked_mul(head_dim)?)?;
                let head_state_end =
                    head_state_start.checked_add(head_dim.checked_mul(head_dim)?)?;
                let head_sum_start = head.checked_mul(head_dim)?;
                let head_sum_end = head_sum_start.checked_add(head_dim)?;

                project_linear_attention_state_i16_checked(
                    &workspace.q[row_base..row_base + head_dim],
                    &workspace.state_kv[head_state_start..head_state_end],
                    &workspace.key_sums[head_sum_start..head_sum_end],
                    head_dim,
                    &mut workspace.context[row_base..row_base + head_dim],
                )?;
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

    Some(())
}

pub fn linear_attention_state_lengths(d_model: usize, heads: usize) -> Option<(usize, usize)> {
    if d_model == 0 || heads == 0 || !d_model.is_multiple_of(heads) {
        return None;
    }

    let head_dim = d_model / heads;
    let state_len = heads.checked_mul(head_dim.checked_mul(head_dim)?)?;
    let key_sum_len = heads.checked_mul(head_dim)?;
    Some((state_len, key_sum_len))
}

pub fn clear_linear_attention_state_checked(
    d_model: usize,
    heads: usize,
    state: LinearAttentionState<'_>,
) -> Option<()> {
    let (state_len, key_sum_len) = linear_attention_state_lengths(d_model, heads)?;
    if state.state_kv.len() < state_len || state.key_sums.len() < key_sum_len {
        return None;
    }

    state.state_kv[..state_len].fill(0);
    state.key_sums[..key_sum_len].fill(0);
    Some(())
}

pub fn linear_attention_step_i16_q15_checked(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionStepWorkspace<'_>,
    state: LinearAttentionState<'_>,
    output_row: &mut [i16],
) -> Option<()> {
    linear_attention_step_i16_q15_with_linear_kernel_checked(
        input_row,
        params,
        workspace,
        state,
        output_row,
        LinearKernel::GenericI8,
    )
}

pub fn linear_attention_step_i16_q15_with_linear_kernel_checked(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionStepWorkspace<'_>,
    state: LinearAttentionState<'_>,
    output_row: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    linear_attention_step_i16_q15_decay_kernel_impl(
        input_row,
        params,
        RESIDUAL_Q15_SCALE,
        workspace,
        state,
        output_row,
        linear_kernel,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_attention_step_i16_q15_decay_kernel_impl(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    decay: FixedScale,
    workspace: LinearAttentionStepWorkspace<'_>,
    state: LinearAttentionState<'_>,
    output_row: &mut [i16],
    linear_kernel: LinearKernel,
) -> Option<()> {
    validate_linear_attention_step_shapes(input_row, params, &workspace, &state, output_row)?;
    if decay.multiplier < 0 || decay.right_shift > MAX_RIGHT_SHIFT {
        return None;
    }

    let d_model = params.d_model;
    let head_dim = d_model / params.heads;

    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.q,
        workspace.q,
        linear_kernel,
    )?;
    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.k,
        workspace.k,
        linear_kernel,
    )?;
    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.v,
        workspace.v,
        linear_kernel,
    )?;

    for head in 0..params.heads {
        let head_offset = head.checked_mul(head_dim)?;
        let head_end = head_offset.checked_add(head_dim)?;
        let head_state_start = head.checked_mul(head_dim.checked_mul(head_dim)?)?;
        let head_state_end = head_state_start.checked_add(head_dim.checked_mul(head_dim)?)?;
        let head_sum_start = head.checked_mul(head_dim)?;
        let head_sum_end = head_sum_start.checked_add(head_dim)?;

        accumulate_linear_attention_state_i16_checked(
            &workspace.k[head_offset..head_end],
            &workspace.v[head_offset..head_end],
            &mut state.state_kv[head_state_start..head_state_end],
            &mut state.key_sums[head_sum_start..head_sum_end],
            head_dim,
            decay,
        )?;
        project_linear_attention_state_i16_checked(
            &workspace.q[head_offset..head_end],
            &state.state_kv[head_state_start..head_state_end],
            &state.key_sums[head_sum_start..head_sum_end],
            head_dim,
            &mut workspace.context[head_offset..head_end],
        )?;
    }

    linear_i16_i8_i16_per_channel_with_kernel_checked(
        workspace.context,
        params.o,
        output_row,
        linear_kernel,
    )?;

    Some(())
}

pub fn linear_attention_ttt_step_i16_q15_checked(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionTttStepWorkspace<'_>,
    state: LinearAttentionState<'_>,
    output_row: &mut [i16],
    learning_rate_shift: u8,
) -> Option<u64> {
    linear_attention_ttt_step_i16_q15_with_linear_kernel_checked(
        input_row,
        params,
        workspace,
        state,
        output_row,
        learning_rate_shift,
        LinearKernel::GenericI8,
    )
}

pub fn linear_attention_ttt_step_i16_q15_with_linear_kernel_checked(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: LinearAttentionTttStepWorkspace<'_>,
    state: LinearAttentionState<'_>,
    output_row: &mut [i16],
    learning_rate_shift: u8,
    linear_kernel: LinearKernel,
) -> Option<u64> {
    validate_linear_attention_ttt_step_shapes(
        input_row,
        params,
        &workspace,
        &state,
        output_row,
        learning_rate_shift,
    )?;

    let d_model = params.d_model;
    let head_dim = d_model / params.heads;

    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.q,
        workspace.q,
        linear_kernel,
    )?;
    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.k,
        workspace.k,
        linear_kernel,
    )?;
    linear_i16_i8_i16_per_channel_with_kernel_checked(
        input_row,
        params.v,
        workspace.v,
        linear_kernel,
    )?;

    let mut delta_l1 = 0_u64;
    for head in 0..params.heads {
        let head_offset = head.checked_mul(head_dim)?;
        let head_end = head_offset.checked_add(head_dim)?;
        let head_state_start = head.checked_mul(head_dim.checked_mul(head_dim)?)?;
        let head_state_end = head_state_start.checked_add(head_dim.checked_mul(head_dim)?)?;
        let head_sum_start = head.checked_mul(head_dim)?;
        let head_sum_end = head_sum_start.checked_add(head_dim)?;

        accumulate_linear_attention_state_i16_checked(
            &workspace.k[head_offset..head_end],
            &workspace.v[head_offset..head_end],
            &mut state.state_kv[head_state_start..head_state_end],
            &mut state.key_sums[head_sum_start..head_sum_end],
            head_dim,
            // TTT manages state plasticity via its own gradient step; no forget gate here.
            RESIDUAL_Q15_SCALE,
        )?;
        let head_delta = linear_attention_ttt_delta_state_i16_q15_checked(
            &workspace.k[head_offset..head_end],
            &workspace.v[head_offset..head_end],
            LinearAttentionState {
                state_kv: &mut state.state_kv[head_state_start..head_state_end],
                key_sums: &mut state.key_sums[head_sum_start..head_sum_end],
            },
            head_dim,
            learning_rate_shift,
            &mut workspace.prediction[head_offset..head_end],
        )?;
        delta_l1 = delta_l1.checked_add(head_delta)?;
        project_linear_attention_state_i16_checked(
            &workspace.q[head_offset..head_end],
            &state.state_kv[head_state_start..head_state_end],
            &state.key_sums[head_sum_start..head_sum_end],
            head_dim,
            &mut workspace.context[head_offset..head_end],
        )?;
    }

    linear_i16_i8_i16_per_channel_with_kernel_checked(
        workspace.context,
        params.o,
        output_row,
        linear_kernel,
    )?;

    Some(delta_l1)
}

/// Apply a denominator-constant test-time-training correction to a linear-attention state.
///
/// The state layout is `[key_dim][value_dim]`, matching the normal linear attention
/// accumulator. `prediction` receives the value predicted by the current state before the
/// update. If the state has no positive normalization denominator yet, the pre-update
/// prediction is treated as zero and the correction still applies.
pub fn linear_attention_ttt_delta_state_i16_q15_checked(
    key: &[i16],
    value: &[i16],
    state: LinearAttentionState<'_>,
    head_dim: usize,
    learning_rate_shift: u8,
    prediction: &mut [i16],
) -> Option<u64> {
    if head_dim == 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
        || key.len() != head_dim
        || value.len() != head_dim
        || prediction.len() != head_dim
        || state.key_sums.len() != head_dim
        || state.state_kv.len() != head_dim.checked_mul(head_dim)?
    {
        return None;
    }

    let mut denominator = 0_i64;
    for (&key_value, &key_sum) in key.iter().zip(state.key_sums.iter()) {
        let product = i64::from(linear_attention_phi_i16_u32(key_value)).checked_mul(key_sum)?;
        denominator = denominator.checked_add(product)?;
    }

    if denominator > 0 {
        for (value_index, out) in prediction.iter_mut().enumerate() {
            let mut numerator = 0_i64;
            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = i64::from(linear_attention_phi_i16_u32(key_value));
                let state_index = key_index.checked_mul(head_dim)?.checked_add(value_index)?;
                let product = phi_key.checked_mul(state.state_kv[state_index])?;
                numerator = numerator.checked_add(product)?;
            }
            *out = saturate_i16(round_div_i64(numerator, denominator));
        }
    } else {
        prediction.fill(0);
    }

    let mut delta_l1 = 0_u64;
    for (key_index, &key_value) in key.iter().enumerate() {
        let phi_key = i64::from(linear_attention_phi_i16_u32(key_value));
        let state_row_start = key_index.checked_mul(head_dim)?;
        for (value_index, (&target_value, &predicted_value)) in
            value.iter().zip(prediction.iter()).enumerate()
        {
            let error = i64::from(target_value) - i64::from(predicted_value);
            let product = phi_key.checked_mul(error)?;
            let update = round_shift_rhu_i64(product, learning_rate_shift);
            let state_index = state_row_start.checked_add(value_index)?;
            state.state_kv[state_index] = state.state_kv[state_index].checked_add(update)?;
            delta_l1 = delta_l1.checked_add(update.unsigned_abs())?;
        }
    }

    Some(delta_l1)
}

#[inline]
fn linear_attention_phi_i16_u32(value: i16) -> u32 {
    // This affine-positive feature map is part of the serialized production
    // model v1 inference contract. A different map requires a new artifact
    // version because model hashes bind bytes, not executable kernel semantics.
    (i32::from(value) + 32769) as u32
}

/// Multiplies the linear-attention recurrent state (`state_kv` = S and
/// `key_sums` = K_s) in place by a fixed-point decay factor γ, using Round-Half-Up.
///
/// γ = `decay.multiplier / 2^decay.right_shift`. Applying the same γ to both S and
/// K_s keeps the projection `output = (φ(Q)·S) / (φ(Q)·K_s)` a valid normalized
/// average — older contributions simply receive geometrically smaller weight.
///
/// A decay of exactly 1 (`RESIDUAL_Q15_SCALE`, i.e. `{multiplier: 1, right_shift: 0}`)
/// is a no-op and returns immediately without touching the state.
pub fn decay_linear_attention_state_i16_checked(
    state_kv: &mut [i64],
    key_sums: &mut [i64],
    head_dim: usize,
    decay: FixedScale,
) -> Option<()> {
    if head_dim == 0
        || key_sums.len() != head_dim
        || state_kv.len() != head_dim.checked_mul(head_dim)?
        || decay.multiplier < 0
        || decay.right_shift > MAX_RIGHT_SHIFT
    {
        return None;
    }

    // Identity decay (γ = 1): nothing to do, and the existing no-decay paths
    // pay zero cost for threading this through.
    if decay.multiplier == 1 && decay.right_shift == 0 {
        return Some(());
    }

    let multiplier = i64::from(decay.multiplier);
    for slot in key_sums.iter_mut() {
        let wide = slot.checked_mul(multiplier)?;
        *slot = round_shift_rhu_i64(wide, decay.right_shift);
    }
    for slot in state_kv.iter_mut() {
        let wide = slot.checked_mul(multiplier)?;
        *slot = round_shift_rhu_i64(wide, decay.right_shift);
    }

    Some(())
}

fn accumulate_linear_attention_state_i16_checked(
    key: &[i16],
    value: &[i16],
    state_kv: &mut [i64],
    key_sums: &mut [i64],
    head_dim: usize,
    decay: FixedScale,
) -> Option<()> {
    if key.len() != head_dim
        || value.len() != head_dim
        || key_sums.len() != head_dim
        || state_kv.len() != head_dim.checked_mul(head_dim)?
    {
        return None;
    }

    // Decay the existing state before folding in this token, so the newest token
    // carries weight γ⁰ and a token k steps in the past carries weight γᵏ.
    decay_linear_attention_state_i16_checked(state_kv, key_sums, head_dim, decay)?;

    for key_index in 0..head_dim {
        let phi_key = i64::from(linear_attention_phi_i16_u32(key[key_index]));
        key_sums[key_index] = key_sums[key_index].checked_add(phi_key)?;
        let state_row_start = key_index.checked_mul(head_dim)?;
        for (value_index, &value) in value.iter().enumerate() {
            let product = phi_key.checked_mul(i64::from(value))?;
            let state_index = state_row_start.checked_add(value_index)?;
            state_kv[state_index] = state_kv[state_index].checked_add(product)?;
        }
    }

    Some(())
}

fn project_linear_attention_state_i16_checked(
    query: &[i16],
    state_kv: &[i64],
    key_sums: &[i64],
    head_dim: usize,
    output: &mut [i16],
) -> Option<()> {
    if query.len() != head_dim
        || output.len() != head_dim
        || key_sums.len() != head_dim
        || state_kv.len() != head_dim.checked_mul(head_dim)?
    {
        return None;
    }

    let mut denominator = 0_i64;
    for (&query, &key_sum) in query.iter().zip(key_sums.iter()) {
        let product = i64::from(linear_attention_phi_i16_u32(query)).checked_mul(key_sum)?;
        denominator = denominator.checked_add(product)?;
    }
    if denominator <= 0 {
        return None;
    }

    for (value_index, out) in output.iter_mut().enumerate() {
        let mut numerator = 0_i64;
        for (key_index, &query) in query.iter().enumerate() {
            let phi_query = i64::from(linear_attention_phi_i16_u32(query));
            let state_index = key_index.checked_mul(head_dim)?.checked_add(value_index)?;
            let product = phi_query.checked_mul(state_kv[state_index])?;
            numerator = numerator.checked_add(product)?;
        }
        *out = saturate_i16(round_div_i64(numerator, denominator));
    }

    Some(())
}

fn round_div_i64(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    let half = denominator >> 1;
    if numerator >= 0 {
        numerator.saturating_add(half) / denominator
    } else {
        numerator.saturating_sub(half) / denominator
    }
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

fn validate_self_attention_shapes(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: &SelfAttentionWorkspace<'_>,
    output: &[i16],
) -> Option<()> {
    if params.seq_len == 0 || params.d_model == 0 || params.heads == 0 {
        return None;
    }

    if !params.d_model.is_multiple_of(params.heads) {
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

fn validate_linear_attention_shapes(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: &LinearAttentionWorkspace<'_>,
    output: &[i16],
) -> Option<()> {
    if params.seq_len == 0 || params.d_model == 0 || params.heads == 0 {
        return None;
    }

    if !params.d_model.is_multiple_of(params.heads) {
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
    let head_dim = params.d_model / params.heads;
    let state_len = params.heads.checked_mul(head_dim.checked_mul(head_dim)?)?;
    let key_sum_len = params.heads.checked_mul(head_dim)?;

    if input.len() != total
        || output.len() != total
        || workspace.q.len() != total
        || workspace.k.len() != total
        || workspace.v.len() != total
        || workspace.context.len() != total
        || workspace.state_kv.len() < state_len
        || workspace.key_sums.len() < key_sum_len
    {
        return None;
    }

    Some(())
}

fn validate_linear_attention_step_shapes(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: &LinearAttentionStepWorkspace<'_>,
    state: &LinearAttentionState<'_>,
    output_row: &[i16],
) -> Option<()> {
    if !params.causal || params.d_model == 0 || params.heads == 0 {
        return None;
    }

    if !params.d_model.is_multiple_of(params.heads) {
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

    let (state_len, key_sum_len) = linear_attention_state_lengths(params.d_model, params.heads)?;
    if input_row.len() != params.d_model
        || output_row.len() != params.d_model
        || workspace.q.len() != params.d_model
        || workspace.k.len() != params.d_model
        || workspace.v.len() != params.d_model
        || workspace.context.len() != params.d_model
        || state.state_kv.len() < state_len
        || state.key_sums.len() < key_sum_len
    {
        return None;
    }

    Some(())
}

fn validate_linear_attention_ttt_step_shapes(
    input_row: &[i16],
    params: SelfAttentionI16Params<'_>,
    workspace: &LinearAttentionTttStepWorkspace<'_>,
    state: &LinearAttentionState<'_>,
    output_row: &[i16],
    learning_rate_shift: u8,
) -> Option<()> {
    if !params.causal
        || params.d_model == 0
        || params.heads == 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return None;
    }

    if !params.d_model.is_multiple_of(params.heads) {
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

    let (state_len, key_sum_len) = linear_attention_state_lengths(params.d_model, params.heads)?;
    if input_row.len() != params.d_model
        || output_row.len() != params.d_model
        || workspace.q.len() != params.d_model
        || workspace.k.len() != params.d_model
        || workspace.v.len() != params.d_model
        || workspace.context.len() != params.d_model
        || workspace.prediction.len() != params.d_model
        || state.state_kv.len() < state_len
        || state.key_sums.len() < key_sum_len
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

    const IDENTITY_2: [i8; 4] = [1, 0, 0, 1];
    const IDENTITY_4: [i8; 16] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

    fn identity_params_2() -> LinearI16I8Params<'static> {
        static SCALES: [crate::FixedScale; 2] = [RESIDUAL_Q15_SCALE; 2];

        LinearI16I8Params {
            weights: &IDENTITY_2,
            bias: None,
            scales: &SCALES,
            input_dim: 2,
            output_dim: 2,
        }
    }

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
    fn q47_reciprocals_track_power_of_two_cases() {
        assert_eq!(reciprocal_sum_q47_lut(1), Some(1_u64 << 47));
        assert_eq!(reciprocal_sum_q47_newton1(2), Some(1_u64 << 46));
        assert_eq!(reciprocal_sum_q47_exact(128), Some(1_u64 << 40));
        assert_eq!(reciprocal_sum_q47_lut(0), None);
        assert_eq!(reciprocal_sum_q47_newton1(0), None);
        assert_eq!(reciprocal_sum_q47_exact(0), None);
    }

    #[test]
    fn q47_newton_refinement_reduces_lut_reciprocal_error() {
        let sum = 8_192_u64 * 32_767;
        let scale = 1_u128 << 47;
        let lut = u128::from(reciprocal_sum_q47_lut(sum).expect("nonzero reciprocal"));
        let newton = u128::from(reciprocal_sum_q47_newton1(sum).expect("nonzero reciprocal"));
        let exact = u128::from(reciprocal_sum_q47_exact(sum).expect("nonzero reciprocal"));
        let mass_error = |reciprocal: u128| {
            u128::from(sum)
                .checked_mul(reciprocal)
                .expect("representative product fits")
                .abs_diff(scale)
        };

        assert!(mass_error(newton) < mass_error(lut));
        assert!(mass_error(exact) <= u128::from(sum / 2));
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
    fn q31_softmax_requantizes_to_the_frozen_q15_result() {
        let logits = [0_i32, -1, -64, -256, MASKED_LOGIT, 19];
        let mut q15 = [0_i16; 6];
        let mut q31 = [0_u32; 6];

        assert_eq!(
            base2_softmax_i32_q15(&logits, &mut q15),
            base2_softmax_i32_q31(&logits, &mut q31)
        );
        for (wide, narrow) in q31.into_iter().zip(q15) {
            assert_eq!(
                saturate_i16(round_shift_rhu_i64(i64::from(wide), 16)),
                narrow
            );
        }
    }

    #[test]
    fn q47_normalization_recovers_uniform_probability_mass() {
        let logits = [0_i32; 8_192];
        let mut legacy = [0_u32; 8_192];
        let mut lut_q47 = [0_u32; 8_192];
        let mut newton_q47 = [0_u32; 8_192];
        let mut exact_q47 = [0_u32; 8_192];

        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut legacy,
            SoftmaxNormalization::LegacyQ31Lut,
        )
        .expect("legacy softmax");
        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut lut_q47,
            SoftmaxNormalization::Q47Lut,
        )
        .expect("Q47 LUT softmax");
        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut newton_q47,
            SoftmaxNormalization::Q47Newton1,
        )
        .expect("Q47 Newton softmax");
        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut exact_q47,
            SoftmaxNormalization::Q47Exact,
        )
        .expect("Q47 exact softmax");

        let scale = 1_u64 << 31;
        let mass_error = |probabilities: &[u32]| {
            probabilities
                .iter()
                .map(|&value| u64::from(value))
                .sum::<u64>()
                .abs_diff(scale)
        };
        assert!(mass_error(&newton_q47) < mass_error(&lut_q47));
        assert!(mass_error(&newton_q47) < mass_error(&legacy));
        assert_eq!(mass_error(&exact_q47), 0);
    }

    #[test]
    fn canonical_integer_nll_is_exact_for_uniform_power_of_two_vocabulary() {
        let logits = [0_i32; 8_192];
        assert_eq!(
            base2_softmax_nll_millibits(&logits, 17, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,),
            Some(13_000)
        );
    }

    #[test]
    fn q47_objective_preserves_small_weights_and_is_target_monotone() {
        assert_eq!(base2_exp_neg_q15(-(16 << 8)), 0);
        assert!(base2_exp_neg_q47(-(16 << 8)) > 0);
        let before = [0_i32, -(20 << 8), -(2 << 8)];
        let mut after = before;
        after[1] += 1;
        let floor = 32_u64 << 20;
        let before_loss = base2_softmax_nll_q47_q20(&before, 1, floor).expect("wide NLL");
        let after_loss = base2_softmax_nll_q47_q20(&after, 1, floor).expect("wide NLL");
        assert!(after_loss <= before_loss);
        let before_q32 = base2_softmax_nll_q47_q32(&before, 1, 32_u64 << 32).expect("wide Q32 NLL");
        let after_q32 = base2_softmax_nll_q47_q32(&after, 1, 32_u64 << 32).expect("wide Q32 NLL");
        assert!(after_q32 < before_q32);
        assert!((before_q32 >> 12).abs_diff(before_loss) <= 1);
        let components = base2_softmax_nll_q47_components(&before, 1).expect("Q47 components");
        assert_eq!(
            components.denominator_log2_q32 >> 12,
            components.denominator_log2_q20
        );
        assert_eq!(components.target_log2_q32 >> 12, components.target_log2_q20);
        assert_eq!(
            components.denominator_log2_q20 - components.target_log2_q20,
            before_loss
        );
        assert_eq!(
            components.denominator_log2_q32 - components.target_log2_q32,
            before_q32
        );
    }

    #[test]
    fn canonical_integer_nll_is_shift_invariant_and_ignores_normalizer_choice() {
        let logits = [512_i32, 128, -64, -512];
        let shifted = [1_536_i32, 1_152, 960, 512];
        let source =
            base2_softmax_nll_millibits(&logits, 2, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS);
        let candidate =
            base2_softmax_nll_millibits(&shifted, 2, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS);
        assert_eq!(source, candidate);

        let mut legacy = [0_u32; 4];
        let mut exact = [0_u32; 4];
        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut legacy,
            SoftmaxNormalization::LegacyQ31Lut,
        )
        .expect("legacy");
        base2_softmax_i32_q31_with_normalization(
            &logits,
            &mut exact,
            SoftmaxNormalization::Q47Exact,
        )
        .expect("exact");
        assert_ne!(legacy, exact);
        assert_eq!(source, Some(2_952));
    }

    #[test]
    fn canonical_integer_nll_uses_declared_floor_for_annihilated_target() {
        let logits = [0_i32, -100_000];
        assert_eq!(
            base2_softmax_nll_millibits(&logits, 1, 29_000),
            Some(29_000)
        );
    }

    #[test]
    fn canonical_integer_log2_q20_matches_frozen_exact_vectors() {
        // Generated once with 120-decimal-digit arithmetic and frozen here so
        // the integer-only repository check never depends on a float oracle.
        for (value, expected_q20) in [
            (1_u64, 0_u64),
            (2, 1_048_576),
            (3, 1_661_953),
            (7, 2_943_724),
            (31, 5_194_851),
            (32_767, 15_728_593),
            (32_768, 15_728_640),
            (268_427_264, 29_360_081),
            (u32::MAX as u64, 33_554_431),
            (u64::MAX, 67_108_863),
        ] {
            let actual = log2_u64_q20(value).expect("positive integer log2");
            assert_eq!(actual, expected_q20, "value={value}");
        }
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
    fn linear_attention_identity_single_token_round_trips_through_projections() {
        let input = [100_i16, -200, 300, -400];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: true,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut state = [0_i64; 16];
        let mut key_sums = [0_i64; 4];
        let workspace = LinearAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            state_kv: &mut state,
            key_sums: &mut key_sums,
        };
        let mut output = [0_i16; 4];

        assert!(linear_attention_i16_q15_checked(&input, params, workspace, &mut output).is_some());
        assert_eq!(output, input);
    }

    #[test]
    fn linear_attention_causal_prefix_blocks_future_value() {
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
        let mut state = [0_i64; 16];
        let mut key_sums = [0_i64; 4];
        let workspace = LinearAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            state_kv: &mut state,
            key_sums: &mut key_sums,
        };
        let mut output = [0_i16; 8];

        assert!(linear_attention_i16_q15_checked(&input, params, workspace, &mut output).is_some());
        assert_eq!(&output[..4], &input[..4]);
        assert_ne!(&output[4..], &input[..4]);
    }

    #[test]
    fn linear_attention_feature_map_preserves_v1_affine_contract() {
        assert_eq!(linear_attention_phi_i16_u32(i16::MIN), 1);
        assert_eq!(linear_attention_phi_i16_u32(-1), 32_768);
        assert_eq!(linear_attention_phi_i16_u32(0), 32_769);
        assert_eq!(linear_attention_phi_i16_u32(1), 32_770);
        assert_eq!(linear_attention_phi_i16_u32(i16::MAX), 65_536);
    }

    #[test]
    fn incremental_linear_attention_matches_full_causal_linear_attention() {
        let input = [
            100_i16, -200, 300, -400, //
            250, 100, -50, 75, //
            -125, 225, 325, -425,
        ];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 3,
            d_model: 4,
            heads: 1,
            causal: true,
        };

        let mut full_q = [0_i16; 12];
        let mut full_k = [0_i16; 12];
        let mut full_v = [0_i16; 12];
        let mut full_context = [0_i16; 12];
        let mut full_state = [0_i64; 16];
        let mut full_key_sums = [0_i64; 4];
        let full_workspace = LinearAttentionWorkspace {
            q: &mut full_q,
            k: &mut full_k,
            v: &mut full_v,
            context: &mut full_context,
            state_kv: &mut full_state,
            key_sums: &mut full_key_sums,
        };
        let mut full_output = [0_i16; 12];
        assert!(
            linear_attention_i16_q15_checked(&input, params, full_workspace, &mut full_output)
                .is_some()
        );

        let mut stream_q = [0_i16; 4];
        let mut stream_k = [0_i16; 4];
        let mut stream_v = [0_i16; 4];
        let mut stream_context = [0_i16; 4];
        let mut stream_state = [0_i64; 16];
        let mut stream_key_sums = [0_i64; 4];
        clear_linear_attention_state_checked(
            4,
            1,
            LinearAttentionState {
                state_kv: &mut stream_state,
                key_sums: &mut stream_key_sums,
            },
        )
        .expect("clear state");
        let mut stream_output = [0_i16; 12];
        for token in 0..3 {
            let row_start = token * 4;
            let row_end = row_start + 4;
            linear_attention_step_i16_q15_checked(
                &input[row_start..row_end],
                params,
                LinearAttentionStepWorkspace {
                    q: &mut stream_q,
                    k: &mut stream_k,
                    v: &mut stream_v,
                    context: &mut stream_context,
                },
                LinearAttentionState {
                    state_kv: &mut stream_state,
                    key_sums: &mut stream_key_sums,
                },
                &mut stream_output[row_start..row_end],
            )
            .expect("streaming step");
        }

        assert_eq!(stream_output, full_output);
    }

    #[test]
    fn linear_attention_ttt_delta_update_teaches_state_current_value() {
        let key = [100_i16, -50];
        let value = [4000_i16, -2000];
        let mut state = [0_i64; 4];
        let mut key_sums = [
            i64::from(linear_attention_phi_i16_u32(key[0])),
            i64::from(linear_attention_phi_i16_u32(key[1])),
        ];
        let mut prediction = [999_i16; 2];

        let delta_l1 = linear_attention_ttt_delta_state_i16_q15_checked(
            &key,
            &value,
            LinearAttentionState {
                state_kv: &mut state,
                key_sums: &mut key_sums,
            },
            2,
            0,
            &mut prediction,
        )
        .expect("ttt update");

        assert_eq!(prediction, [0, 0]);
        assert!(delta_l1 > 0);

        let mut updated_prediction = [0_i16; 2];
        project_linear_attention_state_i16_checked(
            &key,
            &state,
            &key_sums,
            2,
            &mut updated_prediction,
        )
        .expect("updated projection");
        assert_eq!(updated_prediction, value);
    }

    #[test]
    fn linear_attention_ttt_delta_rejects_invalid_shapes() {
        let key = [1_i16, 2];
        let value = [3_i16, 4];
        let mut state = [0_i64; 3];
        let mut key_sums = [1_i64, 1];
        let mut prediction = [0_i16; 2];

        assert_eq!(
            linear_attention_ttt_delta_state_i16_q15_checked(
                &key,
                &value,
                LinearAttentionState {
                    state_kv: &mut state,
                    key_sums: &mut key_sums,
                },
                2,
                0,
                &mut prediction,
            ),
            None
        );
    }

    #[test]
    fn linear_attention_ttt_step_mutates_state_before_projection() {
        let input = [
            100_i16, 0, 0, 0, //
            0, 1000, 0, 0,
        ];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: true,
        };

        let mut normal_q = [0_i16; 4];
        let mut normal_k = [0_i16; 4];
        let mut normal_v = [0_i16; 4];
        let mut normal_context = [0_i16; 4];
        let mut normal_state = [0_i64; 16];
        let mut normal_key_sums = [0_i64; 4];
        let mut normal_output = [0_i16; 8];
        for token in 0..2 {
            let row_start = token * 4;
            let row_end = row_start + 4;
            linear_attention_step_i16_q15_checked(
                &input[row_start..row_end],
                params,
                LinearAttentionStepWorkspace {
                    q: &mut normal_q,
                    k: &mut normal_k,
                    v: &mut normal_v,
                    context: &mut normal_context,
                },
                LinearAttentionState {
                    state_kv: &mut normal_state,
                    key_sums: &mut normal_key_sums,
                },
                &mut normal_output[row_start..row_end],
            )
            .expect("normal stream");
        }

        let mut ttt_q = [0_i16; 4];
        let mut ttt_k = [0_i16; 4];
        let mut ttt_v = [0_i16; 4];
        let mut ttt_context = [0_i16; 4];
        let mut ttt_prediction = [0_i16; 4];
        let mut ttt_state = [0_i64; 16];
        let mut ttt_key_sums = [0_i64; 4];
        let mut ttt_output = [0_i16; 8];
        let mut total_delta_l1 = 0_u64;
        for token in 0..2 {
            let row_start = token * 4;
            let row_end = row_start + 4;
            let delta_l1 = linear_attention_ttt_step_i16_q15_checked(
                &input[row_start..row_end],
                params,
                LinearAttentionTttStepWorkspace {
                    q: &mut ttt_q,
                    k: &mut ttt_k,
                    v: &mut ttt_v,
                    context: &mut ttt_context,
                    prediction: &mut ttt_prediction,
                },
                LinearAttentionState {
                    state_kv: &mut ttt_state,
                    key_sums: &mut ttt_key_sums,
                },
                &mut ttt_output[row_start..row_end],
                0,
            )
            .expect("ttt stream");
            total_delta_l1 = total_delta_l1.checked_add(delta_l1).expect("delta sum");
        }

        assert!(total_delta_l1 > 0);
        assert_ne!(&ttt_state, &normal_state);
        assert_ne!(&ttt_output[4..], &normal_output[4..]);
    }

    #[test]
    fn linear_attention_allows_non_power_of_four_head_dim() {
        let input = [321_i16, -123];
        let params = SelfAttentionI16Params {
            q: identity_params_2(),
            k: identity_params_2(),
            v: identity_params_2(),
            o: identity_params_2(),
            seq_len: 1,
            d_model: 2,
            heads: 1,
            causal: true,
        };
        let mut q = [0_i16; 2];
        let mut k = [0_i16; 2];
        let mut v = [0_i16; 2];
        let mut context = [0_i16; 2];
        let mut state = [0_i64; 4];
        let mut key_sums = [0_i64; 2];
        let workspace = LinearAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            state_kv: &mut state,
            key_sums: &mut key_sums,
        };
        let mut output = [0_i16; 2];

        assert!(linear_attention_i16_q15_checked(&input, params, workspace, &mut output).is_some());
        assert_eq!(output, input);
    }

    #[test]
    fn linear_attention_rejects_too_small_state_workspace() {
        let input = [100_i16, -200, 300, -400];
        let params = SelfAttentionI16Params {
            q: identity_params_4(),
            k: identity_params_4(),
            v: identity_params_4(),
            o: identity_params_4(),
            seq_len: 1,
            d_model: 4,
            heads: 1,
            causal: true,
        };
        let mut q = [0_i16; 4];
        let mut k = [0_i16; 4];
        let mut v = [0_i16; 4];
        let mut context = [0_i16; 4];
        let mut state = [0_i64; 15];
        let mut key_sums = [0_i64; 4];
        let workspace = LinearAttentionWorkspace {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            context: &mut context,
            state_kv: &mut state,
            key_sums: &mut key_sums,
        };
        let mut output = [0_i16; 4];

        assert_eq!(
            linear_attention_i16_q15_checked(&input, params, workspace, &mut output),
            None
        );
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

    #[test]
    fn decay_state_primitive_scales_by_gamma_with_round_half_up() {
        // head_dim = 2 → state_kv has 4 entries, key_sums has 2.
        let mut state_kv = [4_i64, 5, -6, 7];
        let mut key_sums = [10_i64, -3];
        // γ = 1/2 → {multiplier: 1, right_shift: 1}; RhU(x, 1) = (x + 1) >> 1.
        let half = FixedScale {
            multiplier: 1,
            right_shift: 1,
        };

        assert!(
            decay_linear_attention_state_i16_checked(&mut state_kv, &mut key_sums, 2, half)
                .is_some()
        );
        assert_eq!(state_kv, [2, 3, -3, 4]);
        assert_eq!(key_sums, [5, -1]);
    }

    #[test]
    fn decay_state_primitive_identity_is_noop_and_rejects_invalid_input() {
        let mut state_kv = [9_i64, -9, 9, -9];
        let mut key_sums = [123_i64, -456];

        // γ = 1 leaves the state untouched.
        assert!(
            decay_linear_attention_state_i16_checked(
                &mut state_kv,
                &mut key_sums,
                2,
                RESIDUAL_Q15_SCALE,
            )
            .is_some()
        );
        assert_eq!(state_kv, [9, -9, 9, -9]);
        assert_eq!(key_sums, [123, -456]);

        // Wrong state length for head_dim is rejected.
        assert_eq!(
            decay_linear_attention_state_i16_checked(
                &mut [0_i64; 3],
                &mut [0_i64; 2],
                2,
                RESIDUAL_Q15_SCALE,
            ),
            None
        );
        // Negative decay multiplier is rejected.
        assert_eq!(
            decay_linear_attention_state_i16_checked(
                &mut state_kv,
                &mut key_sums,
                2,
                FixedScale {
                    multiplier: -1,
                    right_shift: 0,
                },
            ),
            None
        );
    }
}
