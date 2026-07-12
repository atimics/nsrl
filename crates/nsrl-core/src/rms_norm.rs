use crate::lut::{normalize_u64_to_lut_index, rsqrt_lut_8bit_q15};
use crate::numeric::{round_shift_rhu_i64, saturate_i16};

pub const INV_SQRT_2_Q15: i64 = 23_170;
pub const RMSNORM_INV_RMS_SHIFT: u8 = 30;

pub struct RmsNormBackwardWorkspace<'a> {
    pub normalized_q15: &'a mut [i32],
    pub scaled_grad_q15: &'a mut [i32],
}

pub fn sum_squares_i16_u64_checked(input: &[i16]) -> Option<u64> {
    // Range pre-check: for i16 inputs, each squared term is at most
    // 32768 * 32768 = 1_073_741_824.
    // For n=128 terms, the maximum sum is 137_438_953_472, which fits in u64.
    // Even for n=16384 the max sum is 17_592_186_044_416, still below u64::MAX.
    // So wrapping_add is safe for any realistic dimension.
    // For adversarial sizes (n > u64::MAX / 32768 / 32768) we fall back to checked.
    const MAX_SQUARE: u64 = 32768 * 32768;

    let fits = (input.len() as u128)
        .checked_mul(MAX_SQUARE as u128)
        .is_some_and(|max_sum| max_sum <= u64::MAX as u128);

    if fits {
        // Fast path: use wrapping_add so LLVM can auto-vectorize this loop.
        // Square in i64 (always non-negative), cast to u64, then wrapping-add the u64 sum.
        let mut sum = 0_u64;
        for &value in input {
            let wide = i64::from(value);
            let square = (wide * wide) as u64;
            sum = sum.wrapping_add(square);
        }
        Some(sum)
    } else {
        // Slow path: checked arithmetic for unrealistically large dimensions.
        let mut sum = 0_u64;
        for &value in input {
            let wide = i64::from(value);
            let square = wide.checked_mul(wide)? as u64;
            sum = sum.checked_add(square)?;
        }
        Some(sum)
    }
}

pub fn integer_rsqrt_q30(value: u64) -> Option<i64> {
    let normalized = normalize_u64_to_lut_index(value)?;
    let mut inv = i64::from(rsqrt_lut_8bit_q15(normalized.mantissa)) << 15;
    let exponent = i32::from(normalized.exponent) + 7;

    if exponent & 1 != 0 {
        inv = round_shift_rhu_i64(inv.checked_mul(INV_SQRT_2_Q15)?, 15);
    }

    let half_exponent = exponent.div_euclid(2);
    if half_exponent >= 0 {
        let shift = u32::try_from(half_exponent).ok()?;
        inv = if shift >= 63 { 0 } else { inv >> shift };
    } else {
        let shift = u32::try_from(-half_exponent).ok()?;
        inv = inv.checked_shl(shift)?;
    }

    Some(inv)
}

pub fn rms_norm_i16_q15_checked(
    input: &[i16],
    weights_q15: &[i16],
    eps: u64,
    output: &mut [i16],
) -> Option<()> {
    if input.is_empty() || input.len() != weights_q15.len() || input.len() != output.len() {
        return None;
    }

    let sum_sq = sum_squares_i16_u64_checked(input)?;
    let mean_sq = sum_sq.checked_div(input.len() as u64)?;
    let inv_rms_q30 = integer_rsqrt_q30(mean_sq.checked_add(eps)?)?;

    for ((&value, &weight), out) in input.iter().zip(weights_q15.iter()).zip(output.iter_mut()) {
        let value_weight = i64::from(value).checked_mul(i64::from(weight))?;
        let scaled = value_weight.checked_mul(inv_rms_q30)?;
        *out = saturate_i16(round_shift_rhu_i64(scaled, RMSNORM_INV_RMS_SHIFT));
    }

    Some(())
}

/// Full RMSNorm backward for one row, including the cross-channel RMS term.
///
/// `grad_output_q15` and `grad_input_q15` use Q15. Gamma gradients are
/// accumulated in i64 Q15 so callers can sum deterministically across tokens
/// and batches before applying an optimizer update.
pub fn rms_norm_backward_i16_q15_checked(
    input: &[i16],
    weights_q15: &[i16],
    grad_output_q15: &[i16],
    eps: u64,
    workspace: RmsNormBackwardWorkspace<'_>,
    grad_input_q15: &mut [i16],
    grad_weights_q15: &mut [i64],
) -> Option<usize> {
    if input.is_empty()
        || input.len() != weights_q15.len()
        || input.len() != grad_output_q15.len()
        || input.len() != workspace.normalized_q15.len()
        || input.len() != workspace.scaled_grad_q15.len()
        || input.len() != grad_input_q15.len()
        || input.len() != grad_weights_q15.len()
    {
        return None;
    }
    let sum_sq = sum_squares_i16_u64_checked(input)?;
    let mean_sq = sum_sq.checked_div(input.len() as u64)?;
    let inv_rms_q30 = integer_rsqrt_q30(mean_sq.checked_add(eps)?)?;

    let mut scaled_dot_normalized_q30 = 0_i64;
    for index in 0..input.len() {
        let normalized = round_shift_rhu_i64(i64::from(input[index]).checked_mul(inv_rms_q30)?, 15);
        let normalized = i32::try_from(normalized).ok()?;
        workspace.normalized_q15[index] = normalized;
        let projected = round_shift_rhu_i64(
            i64::from(input[index])
                .checked_mul(i64::from(weights_q15[index]))?
                .checked_mul(inv_rms_q30)?,
            RMSNORM_INV_RMS_SHIFT,
        );
        let active = projected >= i64::from(i16::MIN) && projected <= i64::from(i16::MAX);
        let scaled_grad = if active {
            i32::try_from(round_shift_rhu_i64(
                i64::from(grad_output_q15[index]).checked_mul(i64::from(weights_q15[index]))?,
                15,
            ))
            .ok()?
        } else {
            0
        };
        workspace.scaled_grad_q15[index] = scaled_grad;
        scaled_dot_normalized_q30 = scaled_dot_normalized_q30
            .checked_add(i64::from(scaled_grad).checked_mul(i64::from(normalized))?)?;

        let gamma_grad = if active {
            round_shift_rhu_i64(
                i64::from(grad_output_q15[index]).checked_mul(i64::from(normalized))?,
                15,
            )
        } else {
            0
        };
        grad_weights_q15[index] = grad_weights_q15[index].checked_add(gamma_grad)?;
    }

    let mean_scaled_dot_q15 = round_ratio_i128_rhu(
        i128::from(scaled_dot_normalized_q30),
        i128::try_from(input.len()).ok()?.checked_shl(15)?,
    )?;
    let mean_scaled_dot_q15 = i64::try_from(mean_scaled_dot_q15).ok()?;
    let mut saturation_count = 0_usize;
    for (index, grad_input) in grad_input_q15.iter_mut().enumerate() {
        let cross = round_shift_rhu_i64(
            i64::from(workspace.normalized_q15[index]).checked_mul(mean_scaled_dot_q15)?,
            15,
        );
        let correction = i64::from(workspace.scaled_grad_q15[index]).checked_sub(cross)?;
        let wide = correction.checked_mul(inv_rms_q30)?;
        let grad = round_shift_rhu_i64(wide, RMSNORM_INV_RMS_SHIFT);
        let clamped = saturate_i16(grad);
        if i64::from(clamped) != grad {
            saturation_count = saturation_count.checked_add(1)?;
        }
        *grad_input = clamped;
    }
    Some(saturation_count)
}

fn round_ratio_i128_rhu(numerator: i128, denominator: i128) -> Option<i128> {
    if denominator <= 0 {
        return None;
    }
    numerator
        .checked_add(denominator / 2)
        .map(|value| value.div_euclid(denominator))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_squares_uses_unsigned_large_accumulator() {
        assert_eq!(sum_squares_i16_u64_checked(&[3, -4]), Some(25));
        assert_eq!(
            sum_squares_i16_u64_checked(&[i16::MIN]),
            Some(1_073_741_824)
        );
    }

    #[test]
    fn integer_rsqrt_tracks_power_of_two_inputs() {
        let one = integer_rsqrt_q30(1).expect("rsqrt(1)");
        let four = integer_rsqrt_q30(4).expect("rsqrt(4)");
        let sixteen = integer_rsqrt_q30(16).expect("rsqrt(16)");

        assert!(((1_i64 << 30) - one).abs() < 40_000);
        assert!(((1_i64 << 29) - four).abs() < 20_000);
        assert!(((1_i64 << 28) - sixteen).abs() < 10_000);
        assert_eq!(integer_rsqrt_q30(0), None);
    }

    #[test]
    fn rms_norm_constant_positive_vector_approaches_unit_q15() {
        let input = [1000_i16; 4];
        let weights = [i16::MAX; 4];
        let mut output = [0_i16; 4];

        assert!(rms_norm_i16_q15_checked(&input, &weights, 0, &mut output).is_some());
        assert!(output.iter().all(|&value| value > 32_700));
    }

    #[test]
    fn rms_norm_constant_negative_vector_approaches_negative_unit_q15() {
        let input = [-1000_i16; 4];
        let weights = [i16::MAX; 4];
        let mut output = [0_i16; 4];

        assert!(rms_norm_i16_q15_checked(&input, &weights, 0, &mut output).is_some());
        assert!(output.iter().all(|&value| value < -32_700));
    }

    #[test]
    fn rms_norm_zero_vector_with_eps_stays_zero() {
        let input = [0_i16; 4];
        let weights = [i16::MAX; 4];
        let mut output = [7_i16; 4];

        assert!(rms_norm_i16_q15_checked(&input, &weights, 1, &mut output).is_some());
        assert_eq!(output, [0; 4]);
    }

    #[test]
    fn rms_norm_rejects_bad_shapes_and_zero_rms() {
        let mut output = [0_i16; 2];

        assert_eq!(rms_norm_i16_q15_checked(&[], &[], 1, &mut []), None);
        assert_eq!(
            rms_norm_i16_q15_checked(&[1_i16, 2], &[i16::MAX], 1, &mut output),
            None
        );
        assert_eq!(
            rms_norm_i16_q15_checked(&[0_i16, 0], &[i16::MAX; 2], 0, &mut output),
            None
        );
    }

    #[test]
    fn rms_norm_backward_constant_direction_cancels_cross_channel_term() {
        let input = [1000_i16; 4];
        let weights = [16_384_i16; 4];
        let grad_output = [i16::MAX; 4];
        let mut normalized = [0_i32; 4];
        let mut scaled_grad = [0_i32; 4];
        let mut grad_input = [0_i16; 4];
        let mut grad_weights = [0_i64; 4];

        let saturation = rms_norm_backward_i16_q15_checked(
            &input,
            &weights,
            &grad_output,
            0,
            RmsNormBackwardWorkspace {
                normalized_q15: &mut normalized,
                scaled_grad_q15: &mut scaled_grad,
            },
            &mut grad_input,
            &mut grad_weights,
        )
        .expect("backward");

        assert_eq!(saturation, 0);
        assert!(grad_input.iter().all(|value| value.abs() <= 1));
        assert!(grad_weights.iter().all(|&value| value > 32_700));
    }

    #[test]
    fn rms_norm_backward_matches_precomputed_reference() {
        let input = [300_i16, -700, 1100, 500];
        let weights = [30_000_i16, 27_000, 32_000, 24_000];
        let grad_output = [12_000_i16, -8_000, 5_000, 16_000];
        let mut normalized = [0_i32; 4];
        let mut scaled_grad = [0_i32; 4];
        let mut grad_input = [0_i16; 4];
        let mut grad_weights = [0_i64; 4];

        rms_norm_backward_i16_q15_checked(
            &input,
            &weights,
            &grad_output,
            1,
            RmsNormBackwardWorkspace {
                normalized_q15: &mut normalized,
                scaled_grad_q15: &mut scaled_grad,
            },
            &mut grad_input,
            &mut grad_weights,
        )
        .expect("backward");

        // Rounded from a high-precision reference implementation. Keeping the
        // fixture as integers preserves the repository-wide no-float contract
        // while still locking the cross-channel term and saturation derivative.
        let expected_input = [13_i16, -3, -10, 12];
        let expected_weights = [5_041_i64, 7_842, 0, 11_202];
        for index in 0..input.len() {
            assert!(
                grad_input[index].abs_diff(expected_input[index]) <= 1,
                "input index {index}: actual={} expected={}",
                grad_input[index],
                expected_input[index],
            );
            assert!(
                grad_weights[index].abs_diff(expected_weights[index]) <= 1,
                "gamma index {index}: actual={} expected={}",
                grad_weights[index],
                expected_weights[index],
            );
        }
    }

    #[test]
    fn rms_norm_backward_rejects_shapes_and_handles_extremes() {
        let mut normalized = [0_i32; 2];
        let mut scaled_grad = [0_i32; 2];
        let mut grad_input = [0_i16; 2];
        let mut grad_weights = [0_i64; 2];
        assert_eq!(
            rms_norm_backward_i16_q15_checked(
                &[1_i16],
                &[i16::MAX],
                &[1_i16],
                1,
                RmsNormBackwardWorkspace {
                    normalized_q15: &mut normalized,
                    scaled_grad_q15: &mut scaled_grad,
                },
                &mut grad_input,
                &mut grad_weights,
            ),
            None
        );

        let input = [i16::MIN, i16::MAX];
        let weights = [i16::MAX; 2];
        let grad_output = [i16::MAX, i16::MIN];
        assert!(
            rms_norm_backward_i16_q15_checked(
                &input,
                &weights,
                &grad_output,
                1,
                RmsNormBackwardWorkspace {
                    normalized_q15: &mut normalized,
                    scaled_grad_q15: &mut scaled_grad,
                },
                &mut grad_input,
                &mut grad_weights,
            )
            .is_some()
        );
    }
}
