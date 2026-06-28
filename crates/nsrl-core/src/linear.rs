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
    #[cfg(target_arch = "aarch64")]
    NeonI8,
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
        #[cfg(target_arch = "aarch64")]
        LinearKernel::NeonI8 => linear_i16_i8_i16_per_channel_neon_checked(input, params, output),
    }
}

/// Maximum absolute value of a single i8×i16 product (i8 max = 127, i16 max = 32767).
const MAX_I8_I16_PRODUCT: i64 = 127 * 32767; // = 4_161_409

/// Returns true if the worst-case i8×i16 dot product accumulation provably fits in i32.
///
/// Assumes i8 weights (max |w| = 127) and i16 activations (max |a| = 32767).
/// Worst case: |bias| + input_dim × 127 × 32767 ≤ i32::MAX.
/// The arithmetic is done in i64 to avoid overflow in the check itself.
#[inline]
#[cfg(test)]
pub(crate) fn dot_fits_i32(input_dim: usize, bias: i32) -> bool {
    let max_product_sum = (input_dim as i64).saturating_mul(MAX_I8_I16_PRODUCT);
    let bias_abs = i64::from(bias.unsigned_abs());
    let worst_case = bias_abs.saturating_add(max_product_sum);
    worst_case <= i64::from(i32::MAX)
}

/// Precomputed per-layer overflow bound: input_dim × MAX_I8_I16_PRODUCT.
/// Hoisted out of the tile loop so it isn't recomputed per tile group.
#[inline]
fn max_product_sum_for_dim(input_dim: usize) -> i64 {
    (input_dim as i64).saturating_mul(MAX_I8_I16_PRODUCT)
}

/// Like dot_fits_i32 but uses a precomputed max_product_sum to avoid repeated multiplication.
#[inline]
fn dot_fits_i32_with_sum(max_product_sum: i64, bias: i32) -> bool {
    let bias_abs = i64::from(bias.unsigned_abs());
    bias_abs.saturating_add(max_product_sum) <= i64::from(i32::MAX)
}

/// Fast inner dot-product loop using wrapping arithmetic.
///
/// # Safety (logical)
/// Callers MUST ensure that the true mathematical result fits in i32.
/// Use `dot_fits_i32` to verify this before calling.
#[inline]
fn linear_dot_i16_i8_i32_unchecked(input: &[i16], weights: &[i8], bias_acc: i32) -> i32 {
    let mut acc = bias_acc;
    for (&activation, &weight) in input.iter().zip(weights.iter()) {
        let product = i32::from(activation) * i32::from(weight);
        acc = acc.wrapping_add(product);
    }
    acc
}

/// Tile-4 inner kernel: compute 4 dot products simultaneously over the same input vector.
///
/// `weights` must be exactly 4 × `input_dim` elements stored row-major (rows 0–3 contiguous).
/// This single-slice + stride calling convention is compatible with future AVX2/NEON kernels
/// that operate on a packed contiguous weight block rather than four separate slice pointers.
///
/// LLVM can vectorize all 4 accumulators simultaneously since they share the same `a` value.
///
/// # Safety (logical)
/// Callers MUST verify dot_fits_i32 for each bias before calling.
#[inline]
fn linear_dot4_i16_i8_i32_unchecked(
    input: &[i16],
    weights: &[i8], // 4 × input_dim, row-major
    input_dim: usize,
    b0: i32,
    b1: i32,
    b2: i32,
    b3: i32,
) -> (i32, i32, i32, i32) {
    let w0 = &weights[..input_dim];
    let w1 = &weights[input_dim..2 * input_dim];
    let w2 = &weights[2 * input_dim..3 * input_dim];
    let w3 = &weights[3 * input_dim..];
    let mut acc0 = b0;
    let mut acc1 = b1;
    let mut acc2 = b2;
    let mut acc3 = b3;
    for i in 0..input.len() {
        let a = i32::from(input[i]);
        acc0 = acc0.wrapping_add(a * i32::from(w0[i]));
        acc1 = acc1.wrapping_add(a * i32::from(w1[i]));
        acc2 = acc2.wrapping_add(a * i32::from(w2[i]));
        acc3 = acc3.wrapping_add(a * i32::from(w3[i]));
    }
    (acc0, acc1, acc2, acc3)
}

pub fn linear_i16_i8_i16_per_channel_generic_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
) -> Option<()> {
    if input.len() != params.input_dim || output.len() != params.output_dim || !params.is_valid() {
        return None;
    }

    let input_dim = params.input_dim;
    let output_dim = params.output_dim;
    let weights = params.weights;
    let scales = params.scales;
    let bias = params.bias;

    // Hoist the input_dim-dependent portion of the overflow check out of the tile loop.
    let mps = max_product_sum_for_dim(input_dim);

    // Tile-4 loop: process 4 output rows at a time to allow LLVM to vectorize
    // all 4 accumulators simultaneously while loading the input vector once.
    let mut out_index = 0_usize;
    while out_index + 4 <= output_dim {
        let i0 = out_index;
        let i1 = out_index + 1;
        let i2 = out_index + 2;
        let i3 = out_index + 3;

        let b0 = bias.map_or(0_i32, |b| b[i0]);
        let b1 = bias.map_or(0_i32, |b| b[i1]);
        let b2 = bias.map_or(0_i32, |b| b[i2]);
        let b3 = bias.map_or(0_i32, |b| b[i3]);

        // Check each bias individually so the fast path fires whenever possible.
        let all_fit = dot_fits_i32_with_sum(mps, b0)
            && dot_fits_i32_with_sum(mps, b1)
            && dot_fits_i32_with_sum(mps, b2)
            && dot_fits_i32_with_sum(mps, b3);

        if all_fit {
            // Fast path: tile-4 wrapping kernel over a contiguous 4-row weight block.
            let w_start = i0 * input_dim;
            let (acc0, acc1, acc2, acc3) = linear_dot4_i16_i8_i32_unchecked(
                input,
                &weights[w_start..w_start + 4 * input_dim],
                input_dim,
                b0,
                b1,
                b2,
                b3,
            );
            output[i0] = requantize_i32_to_i16(acc0, scales[i0]);
            output[i1] = requantize_i32_to_i16(acc1, scales[i1]);
            output[i2] = requantize_i32_to_i16(acc2, scales[i2]);
            output[i3] = requantize_i32_to_i16(acc3, scales[i3]);
        } else {
            // Slow path: checked arithmetic for each of the four rows.
            for idx in [i0, i1, i2, i3] {
                let row_start = idx * input_dim;
                let row_end = row_start + input_dim;
                let row_weights = &weights[row_start..row_end];
                let bias_acc = bias.map_or(0_i32, |b| b[idx]);
                let mut acc = bias_acc;
                for (&activation, &weight) in input.iter().zip(row_weights.iter()) {
                    let product = i32::from(activation) * i32::from(weight);
                    acc = acc.checked_add(product)?;
                }
                output[idx] = requantize_i32_to_i16(acc, scales[idx]);
            }
        }

        out_index += 4;
    }

    // Scalar tail: handle remaining output rows when output_dim is not a multiple of 4.
    while out_index < output_dim {
        let row_start = out_index * input_dim;
        let row_end = row_start + input_dim;
        let row_weights = &weights[row_start..row_end];
        let bias_acc = bias.map_or(0_i32, |b| b[out_index]);

        let acc = if dot_fits_i32_with_sum(mps, bias_acc) {
            linear_dot_i16_i8_i32_unchecked(input, row_weights, bias_acc)
        } else {
            let mut acc = bias_acc;
            for (&activation, &weight) in input.iter().zip(row_weights.iter()) {
                let product = i32::from(activation) * i32::from(weight);
                acc = acc.checked_add(product)?;
            }
            acc
        };

        output[out_index] = requantize_i32_to_i16(acc, scales[out_index]);
        out_index += 1;
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

/// NEON tile-4 dispatcher — same structure as the GenericI8 path but calls
/// `neon::linear_dot4_neon_safe` on the fast path.
#[cfg(target_arch = "aarch64")]
pub fn linear_i16_i8_i16_per_channel_neon_checked(
    input: &[i16],
    params: LinearI16I8Params<'_>,
    output: &mut [i16],
) -> Option<()> {
    if input.len() != params.input_dim || output.len() != params.output_dim || !params.is_valid() {
        return None;
    }

    let input_dim = params.input_dim;
    let output_dim = params.output_dim;
    let weights = params.weights;
    let scales = params.scales;
    let bias = params.bias;
    let mps = max_product_sum_for_dim(input_dim);

    let mut out_index = 0_usize;
    while out_index + 4 <= output_dim {
        let i0 = out_index;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let i3 = i0 + 3;
        let b0 = bias.map_or(0_i32, |b| b[i0]);
        let b1 = bias.map_or(0_i32, |b| b[i1]);
        let b2 = bias.map_or(0_i32, |b| b[i2]);
        let b3 = bias.map_or(0_i32, |b| b[i3]);
        let all_fit = dot_fits_i32_with_sum(mps, b0)
            && dot_fits_i32_with_sum(mps, b1)
            && dot_fits_i32_with_sum(mps, b2)
            && dot_fits_i32_with_sum(mps, b3);
        if all_fit {
            let w_start = i0 * input_dim;
            let (acc0, acc1, acc2, acc3) = neon::linear_dot4_neon_safe(
                input,
                &weights[w_start..w_start + 4 * input_dim],
                input_dim,
                b0,
                b1,
                b2,
                b3,
            );
            output[i0] = requantize_i32_to_i16(acc0, scales[i0]);
            output[i1] = requantize_i32_to_i16(acc1, scales[i1]);
            output[i2] = requantize_i32_to_i16(acc2, scales[i2]);
            output[i3] = requantize_i32_to_i16(acc3, scales[i3]);
        } else {
            for idx in [i0, i1, i2, i3] {
                let row_start = idx * input_dim;
                let row_weights = &weights[row_start..row_start + input_dim];
                let bias_acc = bias.map_or(0_i32, |b| b[idx]);
                let mut acc = bias_acc;
                for (&activation, &weight) in input.iter().zip(row_weights.iter()) {
                    let product = i32::from(activation) * i32::from(weight);
                    acc = acc.checked_add(product)?;
                }
                output[idx] = requantize_i32_to_i16(acc, scales[idx]);
            }
        }
        out_index += 4;
    }
    while out_index < output_dim {
        let row_start = out_index * input_dim;
        let row_weights = &weights[row_start..row_start + input_dim];
        let bias_acc = bias.map_or(0_i32, |b| b[out_index]);
        let acc = if dot_fits_i32_with_sum(mps, bias_acc) {
            linear_dot_i16_i8_i32_unchecked(input, row_weights, bias_acc)
        } else {
            let mut acc = bias_acc;
            for (&activation, &weight) in input.iter().zip(row_weights.iter()) {
                acc = acc.checked_add(i32::from(activation) * i32::from(weight))?;
            }
            acc
        };
        output[out_index] = requantize_i32_to_i16(acc, scales[out_index]);
        out_index += 1;
    }
    Some(())
}

mod neon {
    #![allow(unsafe_code)]

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    /// Safe wrapper: caller must have verified via `dot_fits_i32_with_sum` that all four
    /// dot products (including bias) fit in i32, making wrapping arithmetic exact.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(super) fn linear_dot4_neon_safe(
        input: &[i16],
        weights: &[i8], // 4 × input_dim, row-major
        input_dim: usize,
        b0: i32,
        b1: i32,
        b2: i32,
        b3: i32,
    ) -> (i32, i32, i32, i32) {
        // SAFETY: `dot_fits_i32_with_sum` guarantees the true result fits in i32,
        // so wrapping arithmetic produces the correct answer. Slice lengths are
        // correct by construction from `linear_i16_i8_i16_per_channel_neon_checked`.
        unsafe { linear_dot4_neon(input, weights, input_dim, b0, b1, b2, b3) }
    }

    /// Tile-4 NEON kernel: processes 8 input elements per iteration.
    /// `weights` is 4 × input_dim bytes, row-major.
    ///
    /// # Safety
    /// The caller must uphold all of the following; they are guaranteed by
    /// `linear_i16_i8_i16_per_channel_neon_checked`, the only caller:
    /// * `input.len() >= input_dim`. The vector loop reads `input_dim` i16 lanes
    ///   via `vld1q_s16` in 8-lane blocks (`i + 8 <= input_dim`) and the scalar
    ///   tail reads the remainder, so every `acts_ptr.add(i)` for `i < input_dim`
    ///   must be in bounds.
    /// * `weights.len() >= 4 * input_dim`, laid out row-major as four contiguous
    ///   `input_dim`-length rows. Rows are addressed as `w0 = weights`,
    ///   `w1 = w0 + input_dim`, `w2 = w1 + input_dim`, `w3 = w2 + input_dim`, and
    ///   each is read for `i < input_dim`, so `w3.add(input_dim - 1)` must be valid.
    /// * The true mathematical dot product for each of the four rows (including its
    ///   bias) must fit in `i32`. NEON accumulation is wrapping, so this is what
    ///   makes the result exact — verify with `dot_fits_i32_with_sum` beforehand.
    ///
    /// Slices need no special alignment: every load uses the unaligned NEON
    /// variants (`vld1q_s16` / `vld1_s8`). `target_arch = "aarch64"` guarantees the
    /// NEON instructions used here are available (they are part of the baseline ISA).
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn linear_dot4_neon(
        input: &[i16],
        weights: &[i8],
        input_dim: usize,
        b0: i32,
        b1: i32,
        b2: i32,
        b3: i32,
    ) -> (i32, i32, i32, i32) {
        unsafe {
            let acts_ptr = input.as_ptr();
            let w0 = weights.as_ptr();
            let w1 = w0.add(input_dim);
            let w2 = w1.add(input_dim);
            let w3 = w2.add(input_dim);

            let mut acc0 = vdupq_n_s32(0_i32);
            let mut acc1 = vdupq_n_s32(0_i32);
            let mut acc2 = vdupq_n_s32(0_i32);
            let mut acc3 = vdupq_n_s32(0_i32);

            let mut i = 0_usize;
            while i + 8 <= input_dim {
                let acts = vld1q_s16(acts_ptr.add(i));
                let w0_i16 = vmovl_s8(vld1_s8(w0.add(i)));
                acc0 = vmlal_s16(acc0, vget_low_s16(acts), vget_low_s16(w0_i16));
                acc0 = vmlal_high_s16(acc0, acts, w0_i16);
                let w1_i16 = vmovl_s8(vld1_s8(w1.add(i)));
                acc1 = vmlal_s16(acc1, vget_low_s16(acts), vget_low_s16(w1_i16));
                acc1 = vmlal_high_s16(acc1, acts, w1_i16);
                let w2_i16 = vmovl_s8(vld1_s8(w2.add(i)));
                acc2 = vmlal_s16(acc2, vget_low_s16(acts), vget_low_s16(w2_i16));
                acc2 = vmlal_high_s16(acc2, acts, w2_i16);
                let w3_i16 = vmovl_s8(vld1_s8(w3.add(i)));
                acc3 = vmlal_s16(acc3, vget_low_s16(acts), vget_low_s16(w3_i16));
                acc3 = vmlal_high_s16(acc3, acts, w3_i16);
                i += 8;
            }

            let mut r0 = b0.wrapping_add(vaddvq_s32(acc0));
            let mut r1 = b1.wrapping_add(vaddvq_s32(acc1));
            let mut r2 = b2.wrapping_add(vaddvq_s32(acc2));
            let mut r3 = b3.wrapping_add(vaddvq_s32(acc3));

            while i < input_dim {
                let a = i32::from(*acts_ptr.add(i));
                r0 = r0.wrapping_add(a * i32::from(*w0.add(i)));
                r1 = r1.wrapping_add(a * i32::from(*w1.add(i)));
                r2 = r2.wrapping_add(a * i32::from(*w2.add(i)));
                r3 = r3.wrapping_add(a * i32::from(*w3.add(i)));
                i += 1;
            }

            (r0, r1, r2, r3)
        }
    }
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

    const INPUT_TILE: usize = 8;
    let mut input_base = 0_usize;
    while input_base < input_dim {
        let tile_len = (input_dim - input_base).min(INPUT_TILE);
        let mut accs = [0_i64; INPUT_TILE];

        for (out_index, &scaled_grad) in scaled_grad_output.iter().enumerate() {
            let row_start = out_index.checked_mul(input_dim)?.checked_add(input_base)?;
            let row_end = row_start.checked_add(tile_len)?;
            for (tile_index, &weight) in weights[row_start..row_end].iter().enumerate() {
                let product = i64::from(scaled_grad).checked_mul(i64::from(weight))?;
                accs[tile_index] = accs[tile_index].checked_add(product)?;
            }
        }

        for (tile_index, &acc) in accs.iter().enumerate().take(tile_len) {
            let in_index = input_base.checked_add(tile_index)?;
            grad_input[in_index] = requantize_i64_to_i16(acc, grad_input_scales[in_index])?;
        }

        input_base = input_base.checked_add(tile_len)?;
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

    #[test]
    fn dot_fits_i32_accepts_small_dim_and_rejects_overflow() {
        // 128-dim with no bias: 128 × 4_161_409 = 532_660_352, well under i32::MAX
        assert!(dot_fits_i32(128, 0));
        // 515-dim with no bias: 515 × 4_161_409 = 2_143_125_635 < 2_147_483_647 (fits)
        assert!(dot_fits_i32(515, 0));
        // 516-dim with no bias: 516 × 4_161_409 = 2_147_287_044 < 2_147_483_647 (fits)
        assert!(dot_fits_i32(516, 0));
        // 517-dim with no bias: 517 × 4_161_409 = 2_151_448_453 > i32::MAX (overflows)
        assert!(!dot_fits_i32(517, 0));
        // Large bias pushes even small dim over the limit
        assert!(!dot_fits_i32(1, i32::MAX));
        // Zero dim with any bias fits as long as |bias| <= i32::MAX
        assert!(dot_fits_i32(0, i32::MAX));
        assert!(dot_fits_i32(0, i32::MIN + 1));
        // i32::MIN has |bias| = 2_147_483_648 which exceeds i32::MAX
        assert!(!dot_fits_i32(0, i32::MIN));
    }

    #[test]
    fn unchecked_and_checked_paths_produce_identical_output() {
        // Construct a case where dot_fits_i32 is true (fast path is taken),
        // and verify it matches the slow checked path result.
        let input = [100_i16, -200, 300, -400, 500, -600, 127, -127];
        let weights = [2_i8, -3, 4, -5, 6, -7, 8, -9, 1, -1, 1, -1, 1, -1, 1, -1];
        let bias = [42_i32, -42];
        let scales = [RESIDUAL_Q15_SCALE; 2];

        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&bias),
            scales: &scales,
            input_dim: 8,
            output_dim: 2,
        };

        // Verify fast path is taken for this input_dim and bias
        assert!(dot_fits_i32(8, 42));
        assert!(dot_fits_i32(8, -42));

        let mut output_generic = [0_i16; 2];
        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut output_generic)
                .is_some()
        );

        // Also compute manually via the unchecked helper to confirm it matches
        let row0_weights = &weights[..8];
        let row1_weights = &weights[8..16];
        let unchecked_acc0 = linear_dot_i16_i8_i32_unchecked(&input, row0_weights, bias[0]);
        let unchecked_acc1 = linear_dot_i16_i8_i32_unchecked(&input, row1_weights, bias[1]);

        assert_eq!(
            output_generic[0],
            crate::numeric::requantize_i32_to_i16(unchecked_acc0, scales[0])
        );
        assert_eq!(
            output_generic[1],
            crate::numeric::requantize_i32_to_i16(unchecked_acc1, scales[1])
        );
    }

    #[test]
    fn fast_path_matches_slow_path_for_dim_128() {
        // Simulate a realistic layer: input_dim=128, random-ish values.
        // Both paths should agree since 128 easily fits.
        let input: [i16; 128] = core::array::from_fn(|i| (i as i16 * 7 - 500).clamp(-32767, 32767));
        let weights: [i8; 128] =
            core::array::from_fn(|i| ((i as i8).wrapping_mul(3)).clamp(-127, 127));

        // Construct params for a single-output layer
        let bias = [1000_i32; 1];
        let scales = [RESIDUAL_Q15_SCALE; 1];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&bias),
            scales: &scales,
            input_dim: 128,
            output_dim: 1,
        };

        assert!(dot_fits_i32(128, 1000));

        // Run the function (takes fast path due to dot_fits_i32)
        let mut output_fast = [0_i16; 1];
        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut output_fast)
                .is_some()
        );

        // Manually compute expected result using i64 reference arithmetic
        let reference_acc: i64 = input
            .iter()
            .zip(weights.iter())
            .map(|(&a, &w)| i64::from(a) * i64::from(w))
            .sum::<i64>()
            + 1000_i64;
        let reference_i32 = i32::try_from(reference_acc).expect("should fit in i32");
        let expected = crate::numeric::requantize_i32_to_i16(reference_i32, RESIDUAL_Q15_SCALE);

        assert_eq!(output_fast[0], expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernel_matches_generic_i8() {
        // 8 outputs (tile-4 × 2), exactly 8 inputs (one NEON vector)
        let input = [100_i16, -200, 300, -400, 50, -25, 12, -6];
        #[rustfmt::skip]
        let weights = [
            -3_i8, -2, -1, 0, 1, 2, 3, -3,
            -2_i8, -1, 0, 1, 2, 3, -3, -2,
            -1_i8, 0, 1, 2, 3, -3, -2, -1,
             0_i8, 1, 2, 3, -3, -2, -1, 0,
             1_i8, 2, 3, -3, -2, -1, 0, 1,
             2_i8, 3, -3, -2, -1, 0, 1, 2,
             3_i8, -3, -2, -1, 0, 1, 2, 3,
            -3_i8, -2, -1, 0, 1, 2, 3, -3,
        ];
        let scales = [RESIDUAL_Q15_SCALE; 8];
        let bias_vals = [10_i32, -10, 5, -5, 2, -2, 1, -1];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: Some(&bias_vals),
            scales: &scales,
            input_dim: 8,
            output_dim: 8,
        };
        let mut generic_out = [0_i16; 8];
        let mut neon_out = [0_i16; 8];

        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut generic_out)
                .is_some()
        );
        assert!(
            linear_i16_i8_i16_per_channel_neon_checked(&input, params, &mut neon_out).is_some()
        );
        assert_eq!(neon_out, generic_out);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernel_matches_generic_i8_with_tail() {
        // 5 outputs (tile-4 + scalar tail 1), 11 inputs (8 NEON + 3 scalar tail)
        let input = [0_i16, 50, 100, 150, 200, 250, 300, 350, -250, -200, -150];
        #[rustfmt::skip]
        let weights = [
            -2_i8, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2,
            -1_i8,  0, 1, 2, -2, -1, 0, 1, 2, -2, -1,
             0_i8,  1, 2, -2, -1, 0, 1, 2, -2, -1, 0,
             1_i8,  2, -2, -1, 0, 1, 2, -2, -1, 0, 1,
             2_i8, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2,
        ];
        let scales = [RESIDUAL_Q15_SCALE; 5];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 11,
            output_dim: 5,
        };
        let mut generic_out = [0_i16; 5];
        let mut neon_out = [0_i16; 5];

        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut generic_out)
                .is_some()
        );
        assert!(
            linear_i16_i8_i16_per_channel_neon_checked(&input, params, &mut neon_out).is_some()
        );
        assert_eq!(neon_out, generic_out);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernel_matches_generic_i8_exact_multiple_of_8() {
        // 4 outputs (one tile-4), 16 inputs (exactly 2 NEON vectors)
        let input = [
            -800_i16, -700, -600, -500, -400, -300, -200, -100, 0_i16, 100, 200, 300, 400, 500,
            600, 700,
        ];
        #[rustfmt::skip]
        let weights = [
            1_i8, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1,
            0_i8,  1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0,
           -1_i8,  0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1,
            1_i8,  1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1,
        ];
        let scales = [RESIDUAL_Q15_SCALE; 4];
        let params = LinearI16I8Params {
            weights: &weights,
            bias: None,
            scales: &scales,
            input_dim: 16,
            output_dim: 4,
        };
        let mut generic_out = [0_i16; 4];
        let mut neon_out = [0_i16; 4];

        assert!(
            linear_i16_i8_i16_per_channel_generic_checked(&input, params, &mut generic_out)
                .is_some()
        );
        assert!(
            linear_i16_i8_i16_per_channel_neon_checked(&input, params, &mut neon_out).is_some()
        );
        assert_eq!(neon_out, generic_out);
    }
}
