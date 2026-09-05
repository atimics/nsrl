//! Fixed-answer score controls. Targets enter only the scoring function.

pub const ONE: u64 = i16::MAX as u64;

#[derive(Debug, PartialEq, Eq)]
pub struct Scores {
    pub mass: u64,
    pub l1: u64,
    pub brier_numerator: u64,
    pub brier_denominator: u64,
    pub zero_target_probability: bool,
}

pub fn point_mass(predicted: u8) -> [i16; 256] {
    let mut probabilities = [0; 256];
    probabilities[usize::from(predicted)] = i16::MAX;
    probabilities
}

pub fn smoothed_point_mass(predicted: u8) -> [i16; 256] {
    let mut probabilities = [0; 256];
    // Freeze the chosen-byte mass before seeing any target.
    let chosen = 29_491;
    let remaining = i16::MAX - chosen;
    let mut extra = remaining % 255;
    for (index, value) in probabilities.iter_mut().enumerate() {
        if index == usize::from(predicted) {
            *value = chosen;
        } else {
            *value = remaining / 255 + i16::from(extra > 0);
            extra = extra.saturating_sub(1);
        }
    }
    probabilities
}

pub fn score(probabilities: &[i16; 256], target: u8) -> Result<Scores, &'static str> {
    if probabilities.iter().any(|&value| value < 0) {
        return Err("negative probability");
    }
    let mass: u64 = probabilities.iter().map(|&value| value as u64).sum();
    if mass == 0 {
        return Err("zero probability mass");
    }
    let target_mass = probabilities[usize::from(target)] as u64;
    let squares: u64 = probabilities
        .iter()
        .map(|&value| (value as u64) * (value as u64))
        .sum();
    Ok(Scores {
        mass,
        l1: ONE - target_mass + mass - target_mass,
        // Normalize by actual mass to account for Q15 rounding. This is the
        // exact sum of squared distances from the one-hot target.
        brier_numerator: squares + mass * mass - 2 * mass * target_mass,
        brier_denominator: mass * mass,
        zero_target_probability: target_mass == 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_controls_keep_the_answer_and_mass_for_every_byte() {
        for predicted in 0..=u8::MAX {
            for probabilities in [point_mass(predicted), smoothed_point_mass(predicted)] {
                assert_eq!(score(&probabilities, predicted).unwrap().mass, ONE);
                assert!(probabilities.iter().enumerate().all(|(index, &value)| {
                    index == usize::from(predicted) || value < probabilities[usize::from(predicted)]
                }));
            }
        }
    }

    #[test]
    fn one_hot_loss_counts_errors_and_zero_probability_events() {
        let probabilities = point_mass(7);
        let correct = score(&probabilities, 7).unwrap();
        let wrong = score(&probabilities, 8).unwrap();
        assert_eq!(correct.l1, 0);
        assert_eq!(wrong.l1, 2 * ONE);
        assert_eq!(correct.brier_numerator, 0);
        assert_eq!(wrong.brier_numerator, 2 * wrong.brier_denominator);
        assert!(!correct.zero_target_probability);
        assert!(wrong.zero_target_probability);
    }

    #[test]
    fn l1_rewards_forced_certainty_while_brier_prefers_matched_frequency() {
        let mut calibrated = [0; 256];
        calibrated[0] = 24_575;
        calibrated[1] = 8_192;
        let point = point_mass(0);
        let targets = [0, 0, 0, 1];
        let scores = |probabilities: &[i16; 256]| {
            targets.map(|target| score(probabilities, target).unwrap())
        };
        let calibrated_scores = scores(&calibrated);
        let point_scores = scores(&point);
        assert!(
            point_scores.iter().map(|s| s.l1).sum::<u64>()
                < calibrated_scores.iter().map(|s| s.l1).sum::<u64>()
        );
        assert!(
            point_scores.iter().map(|s| s.brier_numerator).sum::<u64>()
                > calibrated_scores
                    .iter()
                    .map(|s| s.brier_numerator)
                    .sum::<u64>()
        );
        assert!(
            calibrated_scores
                .iter()
                .chain(point_scores.iter())
                .all(|s| s.brier_denominator == ONE * ONE)
        );
    }

    #[test]
    fn brier_uses_actual_mass_and_rejects_invalid_input() {
        let mut probabilities = [0; 256];
        assert!(score(&probabilities, 0).is_err());
        probabilities[0] = -1;
        assert!(score(&probabilities, 0).is_err());
        probabilities[0] = 3;
        probabilities[1] = 1;
        let result = score(&probabilities, 0).unwrap();
        assert_eq!(result.brier_numerator, 2);
        assert_eq!(result.brier_denominator, 16);
    }
}
