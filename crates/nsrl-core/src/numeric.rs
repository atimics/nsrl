#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedScale {
    pub multiplier: i32,
    pub right_shift: u8,
}

pub const MAX_RIGHT_SHIFT: u8 = 62;
pub const RESIDUAL_Q15_SCALE: FixedScale = FixedScale {
    multiplier: 1,
    right_shift: 0,
};

pub fn saturate_i8(value: i64) -> i8 {
    value.clamp(i64::from(i8::MIN), i64::from(i8::MAX)) as i8
}

pub fn saturate_i16(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

pub fn saturating_add_i16(left: i16, right: i16) -> i16 {
    left.saturating_add(right)
}

pub fn residual_add_i16_q15_checked(
    skip: &[i16],
    block: &[i16],
    output: &mut [i16],
) -> Option<usize> {
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

pub fn round_shift_rhu_i64(value: i64, right_shift: u8) -> i64 {
    if right_shift == 0 {
        return value;
    }

    debug_assert!(right_shift <= MAX_RIGHT_SHIFT);

    let shift = right_shift.min(MAX_RIGHT_SHIFT);
    let bias = 1_i64 << (shift - 1);
    value.saturating_add(bias) >> shift
}

pub fn requantize_i32_to_i16(accumulator: i32, scale: FixedScale) -> i16 {
    debug_assert!(scale.multiplier >= 0);

    let wide = i64::from(accumulator) * i64::from(scale.multiplier);
    saturate_i16(round_shift_rhu_i64(wide, scale.right_shift))
}

pub fn dot_i8_i16_i32_checked(weights: &[i8], activations: &[i16]) -> Option<i32> {
    if weights.len() != activations.len() {
        return None;
    }

    let mut acc = 0_i32;
    for (&weight, &activation) in weights.iter().zip(activations.iter()) {
        let product = i32::from(weight) * i32::from(activation);
        acc = acc.checked_add(product)?;
    }

    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn saturates_to_integer_bounds() {
        assert_eq!(saturate_i8(i64::from(i8::MIN) - 1), i8::MIN);
        assert_eq!(saturate_i8(i64::from(i8::MAX) + 1), i8::MAX);
        assert_eq!(saturate_i16(i64::from(i16::MIN) - 1), i16::MIN);
        assert_eq!(saturate_i16(i64::from(i16::MAX) + 1), i16::MAX);
    }

    #[test]
    fn round_shift_rhu_matches_locked_signed_bias_behavior() {
        assert_eq!(round_shift_rhu_i64(3, 1), 2);
        assert_eq!(round_shift_rhu_i64(5, 1), 3);
        assert_eq!(round_shift_rhu_i64(-3, 1), -1);
        assert_eq!(round_shift_rhu_i64(-5, 1), -2);
        assert_eq!(round_shift_rhu_i64(123, 0), 123);
    }

    #[test]
    fn requantize_saturates_after_scaling() {
        let scale = FixedScale {
            multiplier: 1000,
            right_shift: 0,
        };

        assert_eq!(requantize_i32_to_i16(100, scale), i16::MAX);
        assert_eq!(requantize_i32_to_i16(-100, scale), i16::MIN);
    }

    #[test]
    fn residual_add_counts_saturations_and_rejects_shape_mismatch() {
        let skip = [32_000_i16, -32_000, 100];
        let block = [1_000_i16, -1_000, -25];
        let mut output = [0_i16; 3];

        assert_eq!(
            residual_add_i16_q15_checked(&skip, &block, &mut output),
            Some(2)
        );
        assert_eq!(output, [i16::MAX, i16::MIN, 75]);
        assert_eq!(
            residual_add_i16_q15_checked(&skip[..2], &block, &mut output),
            None
        );
    }

    #[test]
    fn checked_dot_rejects_mismatched_lengths_and_overflow() {
        assert_eq!(dot_i8_i16_i32_checked(&[1, 2], &[3]), None);

        let weights = [127_i8; 600];
        let activations = [i16::MAX; 600];
        assert_eq!(dot_i8_i16_i32_checked(&weights, &activations), None);
    }

    #[test]
    fn checked_dot_computes_small_vectors() {
        let weights = [2_i8, -3, 4];
        let activations = [10_i16, 20, -30];

        assert_eq!(dot_i8_i16_i32_checked(&weights, &activations), Some(-160));
    }

    proptest! {
        #[test]
        fn requantize_i32_to_i16_never_panics(
            acc in any::<i32>(),
            multiplier in 0_i32..=i32::MAX,
            right_shift in 0_u8..=MAX_RIGHT_SHIFT,
        ) {
            let scale = FixedScale { multiplier, right_shift };
            let _ = requantize_i32_to_i16(acc, scale);
        }

        #[test]
        fn checked_dot_matches_i64_reference_when_it_fits(
            weights in proptest::collection::vec(any::<i8>(), 0..256),
            activations in proptest::collection::vec(any::<i16>(), 0..256),
        ) {
            let len = core::cmp::min(weights.len(), activations.len());
            let weights = &weights[..len];
            let activations = &activations[..len];

            let reference = weights.iter()
                .zip(activations.iter())
                .map(|(&w, &a)| i64::from(w) * i64::from(a))
                .sum::<i64>();

            let got = dot_i8_i16_i32_checked(weights, activations);

            if let Ok(reference_i32) = i32::try_from(reference) {
                prop_assert_eq!(got, Some(reference_i32));
            } else {
                prop_assert_eq!(got, None);
            }
        }
    }
}
