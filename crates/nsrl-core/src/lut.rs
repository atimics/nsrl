include!(concat!(env!("OUT_DIR"), "/rsqrt_lut_8bit.rs"));

pub const RSQRT_LUT_8BIT_DOMAIN_START: u8 = 128;
pub const RSQRT_LUT_8BIT_DOMAIN_END: u8 = 255;
pub const RSQRT_LUT_8BIT_LEN: usize = 128;
pub const EXP2_NEG_FRAC_LUT_8BIT_LEN: usize = 256;
pub const RECIP_LUT_8BIT_Q31_LEN: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizedMantissa {
    pub mantissa: u8,
    pub exponent: i16,
    pub lut_index: usize,
}

pub fn normalize_u64_to_lut_index(value: u64) -> Option<NormalizedMantissa> {
    if value == 0 {
        return None;
    }

    let msb = 63_u32 - value.leading_zeros();
    let (mantissa, exponent) = if msb >= 7 {
        let shift = msb - 7;
        ((value >> shift) as u8, shift as i16)
    } else {
        let shift = 7 - msb;
        ((value << shift) as u8, -(shift as i16))
    };

    debug_assert!((RSQRT_LUT_8BIT_DOMAIN_START..=RSQRT_LUT_8BIT_DOMAIN_END).contains(&mantissa));

    Some(NormalizedMantissa {
        mantissa,
        exponent,
        lut_index: usize::from(mantissa - RSQRT_LUT_8BIT_DOMAIN_START),
    })
}

pub fn rsqrt_lut_8bit_q15(mantissa: u8) -> i16 {
    assert!(
        (RSQRT_LUT_8BIT_DOMAIN_START..=RSQRT_LUT_8BIT_DOMAIN_END).contains(&mantissa),
        "mantissa must be normalized into [128, 255]"
    );
    RSQRT_LUT_8BIT[usize::from(mantissa - RSQRT_LUT_8BIT_DOMAIN_START)]
}

pub fn recip_lut_8bit_q31(mantissa: u8) -> u32 {
    assert!(
        (RSQRT_LUT_8BIT_DOMAIN_START..=RSQRT_LUT_8BIT_DOMAIN_END).contains(&mantissa),
        "mantissa must be normalized into [128, 255]"
    );
    RECIP_LUT_8BIT_Q31[usize::from(mantissa - RSQRT_LUT_8BIT_DOMAIN_START)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_nonzero_values_into_8bit_window() {
        let cases = [
            (1_u64, 128_u8, -7_i16),
            (2, 128, -6),
            (127, 254, -1),
            (128, 128, 0),
            (255, 255, 0),
            (256, 128, 1),
            (u64::MAX, 255, 56),
        ];

        for (value, mantissa, exponent) in cases {
            let normalized = normalize_u64_to_lut_index(value).expect("nonzero value");
            assert_eq!(normalized.mantissa, mantissa);
            assert_eq!(normalized.exponent, exponent);
            assert_eq!(
                normalized.lut_index,
                usize::from(mantissa - RSQRT_LUT_8BIT_DOMAIN_START)
            );
        }
    }

    #[test]
    fn zero_has_no_normalized_mantissa() {
        assert_eq!(normalize_u64_to_lut_index(0), None);
    }

    #[test]
    fn rsqrt_lut_is_q15_and_monotonic_decreasing() {
        assert_eq!(RSQRT_LUT_8BIT.len(), RSQRT_LUT_8BIT_LEN);
        assert_eq!(rsqrt_lut_8bit_q15(128), i16::MAX);

        for pair in RSQRT_LUT_8BIT.windows(2) {
            assert!(pair[0] >= pair[1]);
            assert!(pair[1] > 0);
        }
    }

    #[test]
    fn exp2_negative_fraction_lut_is_q15_and_monotonic_decreasing() {
        assert_eq!(EXP2_NEG_FRAC_LUT_8BIT.len(), EXP2_NEG_FRAC_LUT_8BIT_LEN);
        assert_eq!(EXP2_NEG_FRAC_LUT_8BIT[0], i16::MAX);

        for pair in EXP2_NEG_FRAC_LUT_8BIT.windows(2) {
            assert!(pair[0] >= pair[1]);
            assert!(pair[1] > 0);
        }
    }

    #[test]
    fn reciprocal_lut_is_q31_and_monotonic_decreasing() {
        assert_eq!(RECIP_LUT_8BIT_Q31.len(), RECIP_LUT_8BIT_Q31_LEN);
        assert_eq!(recip_lut_8bit_q31(128), 1_u32 << 24);

        for pair in RECIP_LUT_8BIT_Q31.windows(2) {
            assert!(pair[0] >= pair[1]);
            assert!(pair[1] > 0);
        }
    }
}
