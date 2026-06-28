//! Integer-only dynamic-range calibration for `FixedScale` requantization.
//!
//! The runtime is deterministic and float-free, so scale calibration is too. The
//! pipeline is:
//!
//! 1. [`MagnitudeHistogram`] accumulates the absolute magnitudes of observed
//!    values (accumulators, activations) into power-of-two buckets during a
//!    forward pass — O(1) per observation, no allocation, `no_std`.
//! 2. [`MagnitudeHistogram::coverage_magnitude`] returns the magnitude bound that
//!    covers a requested fraction of observations, so a few outliers can be
//!    allowed to clip rather than forcing the whole range to shrink.
//! 3. [`solve_fixed_scale_ratio`] turns a desired ratio `numerator / denominator`
//!    into the best `FixedScale { multiplier, right_shift }` using i128 integer
//!    math, maximizing precision while keeping the multiplier inside `i32`.
//!
//! [`calibrate_fixed_scale`] composes these into a one-call "map this layer's
//! observed accumulator range into the i16 output range, tolerating `1 - coverage`
//! clipping" helper suitable for a checkpoint-time calibration pass.

use crate::numeric::{FixedScale, MAX_RIGHT_SHIFT};

/// Number of power-of-two magnitude buckets: bit-lengths `0..=64`.
pub const MAGNITUDE_BUCKETS: usize = 65;

/// Bit-length of `value`: `0` for `0`, otherwise `floor(log2(value)) + 1`.
///
/// This is the magnitude-bucket index used throughout the histogram.
#[inline]
pub fn bit_length_u64(value: u64) -> usize {
    (64 - value.leading_zeros()) as usize
}

/// Running histogram of observed magnitudes, bucketed by bit-length.
///
/// Bucket `b` counts observations whose absolute value has bit-length `b`
/// (i.e. lies in `[2^(b-1), 2^b)`, with bucket `0` counting exact zeros).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MagnitudeHistogram {
    counts: [u64; MAGNITUDE_BUCKETS],
    total: u64,
    max_abs: u64,
}

impl Default for MagnitudeHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl MagnitudeHistogram {
    pub const fn new() -> Self {
        Self {
            counts: [0; MAGNITUDE_BUCKETS],
            total: 0,
            max_abs: 0,
        }
    }

    /// Records one observation by its absolute magnitude. Saturates the bucket
    /// count and total at `u64::MAX` rather than wrapping, so a pathologically
    /// long calibration run degrades gracefully instead of corrupting the tally.
    #[inline]
    pub fn observe_magnitude(&mut self, magnitude: u64) {
        let bucket = bit_length_u64(magnitude);
        self.counts[bucket] = self.counts[bucket].saturating_add(1);
        self.total = self.total.saturating_add(1);
        if magnitude > self.max_abs {
            self.max_abs = magnitude;
        }
    }

    #[inline]
    pub fn observe_i64(&mut self, value: i64) {
        self.observe_magnitude(value.unsigned_abs());
    }

    pub fn observe_i32_slice(&mut self, values: &[i32]) {
        for &value in values {
            self.observe_i64(i64::from(value));
        }
    }

    pub fn observe_i16_slice(&mut self, values: &[i16]) {
        for &value in values {
            self.observe_i64(i64::from(value));
        }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn max_abs(&self) -> u64 {
        self.max_abs
    }

    pub fn counts(&self) -> &[u64; MAGNITUDE_BUCKETS] {
        &self.counts
    }

    /// Merges another histogram into this one (e.g. reducing per-shard tallies
    /// from a training swarm). Counts and totals saturate at `u64::MAX`.
    pub fn merge(&mut self, other: &MagnitudeHistogram) {
        for (slot, &add) in self.counts.iter_mut().zip(other.counts.iter()) {
            *slot = slot.saturating_add(add);
        }
        self.total = self.total.saturating_add(other.total);
        if other.max_abs > self.max_abs {
            self.max_abs = other.max_abs;
        }
    }

    /// Smallest power-of-two magnitude bound `2^b` such that at least
    /// `coverage_num / coverage_den` of all observations are `< 2^b` (i.e. fall in
    /// buckets `0..=b`). Returns `None` if no observations have been recorded or if
    /// `coverage_num > coverage_den` (an impossible coverage request).
    ///
    /// The result is a conservative (upper) bound on the covered magnitudes:
    /// choosing the bucket's upper edge `2^b` guarantees the coverage target is met.
    pub fn coverage_magnitude(&self, coverage_num: u64, coverage_den: u64) -> Option<u64> {
        if self.total == 0 || coverage_den == 0 || coverage_num > coverage_den {
            return None;
        }

        // Target count = ceil(total * coverage_num / coverage_den), in u128 to avoid
        // overflow, so the returned bound covers *at least* the requested fraction.
        let scaled = u128::from(self.total) * u128::from(coverage_num);
        let target = scaled.div_ceil(u128::from(coverage_den));

        let mut cumulative = 0_u128;
        for (bucket, &count) in self.counts.iter().enumerate() {
            cumulative += u128::from(count);
            if cumulative >= target {
                // Upper edge of this bucket: bucket 0 (zeros) covers magnitude 0;
                // bucket b covers up to 2^b - 1, so the conservative bound is 2^b.
                return Some(if bucket == 0 { 0 } else { 1_u64 << (bucket - 1) << 1 });
            }
        }

        // Coverage target not reached within recorded buckets (only possible under
        // saturation); fall back to the observed maximum.
        Some(self.max_abs)
    }
}

/// Solves for the `FixedScale { multiplier, right_shift }` that best approximates
/// `numerator / denominator` using integer-only arithmetic.
///
/// Picks the largest `right_shift` (≤ [`MAX_RIGHT_SHIFT`]) for which the multiplier
/// still fits in `i32`, maximizing fractional precision. The multiplier is rounded
/// half-up. Behavior at the edges:
///
/// - `denominator == 0` → `None` (no observed range to calibrate against).
/// - `numerator == 0` → `Some({0, 0})` (collapses everything to zero).
/// - ratio larger than `i32::MAX` → `Some({i32::MAX, 0})` (largest representable scale).
pub fn solve_fixed_scale_ratio(numerator: u64, denominator: u64) -> Option<FixedScale> {
    if denominator == 0 {
        return None;
    }
    if numerator == 0 {
        return Some(FixedScale {
            multiplier: 0,
            right_shift: 0,
        });
    }

    let num = u128::from(numerator);
    let den = u128::from(denominator);
    let max_multiplier = i32::MAX as u128;
    let half = den / 2;

    // multiplier(s) = round(numerator * 2^s / denominator) is monotonic in s, so
    // scanning from the largest shift down returns the highest-precision scale
    // whose multiplier still fits in i32.
    let mut shift = MAX_RIGHT_SHIFT;
    loop {
        let multiplier = ((num << shift) + half) / den;
        if multiplier <= max_multiplier {
            return Some(FixedScale {
                multiplier: multiplier as i32,
                right_shift: shift,
            });
        }
        if shift == 0 {
            // Even at shift 0 the ratio exceeds i32::MAX: clamp to the largest
            // representable scale.
            return Some(FixedScale {
                multiplier: i32::MAX,
                right_shift: 0,
            });
        }
        shift -= 1;
    }
}

/// Derives a requantization scale that maps a layer's observed magnitudes into a
/// target output range, allowing `1 - coverage_num/coverage_den` of observations
/// to clip.
///
/// `target_output_max` is the magnitude the covered range should map to — typically
/// `i16::MAX as u64` (32767) for an i16 activation output. Returns `None` if the
/// histogram is empty, the coverage request is invalid, or the covered magnitude is
/// zero (no usable signal).
pub fn calibrate_fixed_scale(
    histogram: &MagnitudeHistogram,
    target_output_max: u64,
    coverage_num: u64,
    coverage_den: u64,
) -> Option<FixedScale> {
    let bound = histogram.coverage_magnitude(coverage_num, coverage_den)?;
    if bound == 0 {
        return None;
    }
    solve_fixed_scale_ratio(target_output_max, bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::requantize_i32_to_i16;

    #[test]
    fn bit_length_matches_definition() {
        assert_eq!(bit_length_u64(0), 0);
        assert_eq!(bit_length_u64(1), 1);
        assert_eq!(bit_length_u64(2), 2);
        assert_eq!(bit_length_u64(3), 2);
        assert_eq!(bit_length_u64(4), 3);
        assert_eq!(bit_length_u64(255), 8);
        assert_eq!(bit_length_u64(256), 9);
        assert_eq!(bit_length_u64(u64::MAX), 64);
    }

    #[test]
    fn histogram_tracks_total_and_max() {
        let mut hist = MagnitudeHistogram::new();
        hist.observe_i32_slice(&[10, -20, 300, -4]);
        assert_eq!(hist.total(), 4);
        assert_eq!(hist.max_abs(), 300);
        // 300 has bit-length 9 (256..511).
        assert_eq!(hist.counts()[9], 1);
        // 10 -> bit-length 4, 20 -> 5, 4 -> 3.
        assert_eq!(hist.counts()[4], 1);
        assert_eq!(hist.counts()[5], 1);
        assert_eq!(hist.counts()[3], 1);
    }

    #[test]
    fn coverage_magnitude_picks_bucket_upper_edge() {
        let mut hist = MagnitudeHistogram::new();
        // 99 small values (magnitude 5 -> bit-length 3) and 1 large outlier (4096).
        for _ in 0..99 {
            hist.observe_i64(5);
        }
        hist.observe_i64(4096); // bit-length 13

        // 99% coverage should exclude the outlier and bound at 2^3 = 8.
        assert_eq!(hist.coverage_magnitude(99, 100), Some(8));
        // 100% coverage must include the outlier: 4096 has bit-length 13 -> 2^13.
        assert_eq!(hist.coverage_magnitude(100, 100), Some(1 << 13));
    }

    #[test]
    fn coverage_rejects_empty_or_impossible_requests() {
        let empty = MagnitudeHistogram::new();
        assert_eq!(empty.coverage_magnitude(1, 1), None);

        let mut hist = MagnitudeHistogram::new();
        hist.observe_i64(1);
        assert_eq!(hist.coverage_magnitude(2, 1), None); // coverage > 100%
        assert_eq!(hist.coverage_magnitude(1, 0), None); // zero denominator
    }

    #[test]
    fn merge_combines_counts_and_max() {
        let mut a = MagnitudeHistogram::new();
        a.observe_i64(3);
        let mut b = MagnitudeHistogram::new();
        b.observe_i64(1000);
        a.merge(&b);
        assert_eq!(a.total(), 2);
        assert_eq!(a.max_abs(), 1000);
    }

    #[test]
    fn solve_ratio_represents_unity_at_max_precision() {
        // 1/1: largest shift whose multiplier (2^s) fits in i32 is 30 (2^30 ≤ i32::MAX < 2^31).
        let scale = solve_fixed_scale_ratio(1, 1).expect("scale");
        assert_eq!(
            scale,
            FixedScale {
                multiplier: 1 << 30,
                right_shift: 30,
            }
        );
    }

    #[test]
    fn solve_ratio_represents_one_half() {
        let scale = solve_fixed_scale_ratio(1, 2).expect("scale");
        assert_eq!(
            scale,
            FixedScale {
                multiplier: 1 << 30,
                right_shift: 31,
            }
        );
    }

    #[test]
    fn solve_ratio_edge_cases() {
        assert_eq!(solve_fixed_scale_ratio(5, 0), None);
        assert_eq!(
            solve_fixed_scale_ratio(0, 5),
            Some(FixedScale {
                multiplier: 0,
                right_shift: 0,
            })
        );
        // Ratio far exceeding i32 range clamps to the largest representable scale.
        assert_eq!(
            solve_fixed_scale_ratio(u64::MAX, 1),
            Some(FixedScale {
                multiplier: i32::MAX,
                right_shift: 0,
            })
        );
    }

    #[test]
    fn solved_scale_maps_observed_max_near_target() {
        // An accumulator that peaks around 1_000_000 should requantize to ~32767.
        let scale = solve_fixed_scale_ratio(32767, 1_000_000).expect("scale");
        let mapped = requantize_i32_to_i16(1_000_000, scale);
        // Within 0.1% of the target output magnitude.
        assert!((mapped as i32 - 32767).abs() <= 33, "mapped = {mapped}");
        // And a value inside the range never overflows i16.
        let _ = requantize_i32_to_i16(999_999, scale);
    }

    #[test]
    fn calibrate_end_to_end_keeps_covered_range_in_bounds() {
        let mut hist = MagnitudeHistogram::new();
        // Bulk of accumulators near ±50_000, with a handful of large outliers.
        for i in 0..1000 {
            let v = 40_000 + (i % 20_000);
            hist.observe_i32_slice(&[v, -v]);
        }
        hist.observe_i32_slice(&[5_000_000]); // lone outlier to be clipped

        let scale = calibrate_fixed_scale(&hist, i16::MAX as u64, 999, 1000).expect("calibrated");

        // The 99.9% coverage bound (~2^16) maps comfortably inside i16 range, while
        // the rare 5M outlier is allowed to saturate rather than crushing the scale.
        let bound = hist.coverage_magnitude(999, 1000).expect("bound");
        let mapped_bound = requantize_i32_to_i16(bound as i32, scale);
        assert!(mapped_bound > 0);
        assert!(bound < 5_000_000, "outliers should be excluded: {bound}");
    }
}
