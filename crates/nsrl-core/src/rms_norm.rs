use crate::lut::{normalize_u64_to_lut_index, rsqrt_lut_8bit_q15};
use crate::numeric::{round_shift_rhu_i64, saturate_i16};

pub const INV_SQRT_2_Q15: i64 = 23_170;
pub const RMSNORM_INV_RMS_SHIFT: u8 = 30;

pub fn sum_squares_i16_u64_checked(input: &[i16]) -> Option<u64> {
    // Range pre-check: for i16 inputs, each squared term is at most 32767² = 1_073_692_169.
    // For n=128 terms, the maximum sum is 128 × 1_073_692_169 ≈ 137.4B, which fits in u64.
    // Even for n=16384 the max sum ≈ 17.6T still fits in u64 (max ~1.84×10¹⁹).
    // So wrapping_add is safe for any realistic dimension.
    // For adversarial sizes (n > u64::MAX / 32767²) we fall back to checked.
    const MAX_SQUARE: u64 = 32767 * 32767; // = 1_073_692_289

    let fits = (input.len() as u128)
        .checked_mul(MAX_SQUARE as u128)
        .map_or(false, |max_sum| max_sum <= u64::MAX as u128);

    if fits {
        // Fast path: use wrapping_add so LLVM can auto-vectorize this loop.
        // Square in i64 (always non-negative), cast to u64, then wrapping-add the u64 sum.
        let mut sum = 0_u64;
        for &value in input {
            let wide = i64::from(value);
            let square = (wide * wide) as u64; // wide*wide >= 0, fits in u64 (max 32767²)
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
}
