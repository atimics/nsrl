use std::fmt::Write;

use nsrl_core::MAX_RIGHT_SHIFT;

use super::{ProductionFullTrainConfig, ProductionModelConfig, ProductionProjectionScales};

pub const PRODUCTION_ACTIVATION_FRACTIONAL_BITS: i16 = 15;
pub const PRODUCTION_LOGIT_FRACTIONAL_BITS: i16 = 8;
pub const PRODUCTION_RMS_SQUARE_FRACTIONAL_BITS: i16 = 30;

const MAX_I8_MAGNITUDE: u128 = 1_u128 << 7;
const MAX_I16_MAGNITUDE: u128 = 1_u128 << 15;
const LINEAR_ATTENTION_PHI_MAX: u128 = 1_u128 << 16;
const TRAINABLE_GROUP_NAMES: [&str; 13] = [
    "embeddings",
    "attention_rms",
    "mlp_rms",
    "final_rms",
    "q",
    "k",
    "v",
    "o",
    "up",
    "gate",
    "down",
    "output",
    "bias",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionRoundingRule {
    RoundHalfUp,
    ArithmeticRightShift,
    NearestTiesAwayFromZero,
}

impl ProductionRoundingRule {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoundHalfUp => "round_half_up_toward_positive_infinity",
            Self::ArithmeticRightShift => "arithmetic_right_shift_toward_negative_infinity",
            Self::NearestTiesAwayFromZero => "nearest_ties_away_from_zero",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionProjectionContract {
    pub name: &'static str,
    pub input_fractional_bits: i16,
    pub weight_fractional_bits: i16,
    pub accumulator_fractional_bits: i16,
    pub output_fractional_bits: i16,
    pub right_shift: u8,
    pub rounding: ProductionRoundingRule,
    pub maximum_absolute_accumulator: u64,
    pub accumulator_bits_required: u8,
}

impl ProductionProjectionContract {
    pub fn is_dimensionally_consistent(&self) -> bool {
        self.accumulator_fractional_bits == self.input_fractional_bits + self.weight_fractional_bits
            && self.output_fractional_bits
                == self.accumulator_fractional_bits - i16::from(self.right_shift)
            && self.maximum_absolute_accumulator <= i64::MAX as u64
            && self.accumulator_bits_required <= 64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionAttentionBounds {
    pub maximum_key_sum: u64,
    pub maximum_absolute_state_value: u64,
    pub maximum_denominator: u64,
    pub maximum_absolute_numerator: u64,
    pub division_rounding: ProductionRoundingRule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionNumericContract {
    pub profile: &'static str,
    pub activation_fractional_bits: i16,
    pub logit_fractional_bits: i16,
    pub rms_square_fractional_bits: i16,
    pub rms_epsilon_stored_units: u64,
    pub projections: Vec<ProductionProjectionContract>,
    pub attention: ProductionAttentionBounds,
    pub maximum_rms_sum_squares: u64,
    pub maximum_softmax_weight_sum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBackwardEdgeContract {
    pub name: &'static str,
    pub input_gradient_fractional_bits: i16,
    pub weight_fractional_bits: i16,
    pub accumulator_fractional_bits: i16,
    pub right_shift: u8,
    pub arithmetic_output_fractional_bits: i16,
    pub interpreted_output_fractional_bits: i16,
    pub interpretation_delta_bits: i16,
    pub rounding: ProductionRoundingRule,
    pub straight_through_nonzero_rescue: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionParameterUpdateContract {
    pub name: &'static str,
    pub parameter_fractional_bits: i16,
    pub raw_gradient_fractional_bits: i16,
    pub update_right_shift: u8,
    pub real_learning_rate_power_of_two: i16,
    pub rounding: ProductionRoundingRule,
    pub residual_error_feedback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionTrainingNumericContract {
    pub profile: &'static str,
    pub probability_gradient_fractional_bits: u8,
    pub internal_gradient_fractional_bits: i16,
    pub output_backward_base_shift: u8,
    pub output_backward_effective_shift: u8,
    pub attention_context_gradient_left_shift: u8,
    pub backward_edges: Vec<ProductionBackwardEdgeContract>,
    pub parameter_updates: Vec<ProductionParameterUpdateContract>,
}

impl ProductionTrainingNumericContract {
    pub fn derive(
        config: ProductionModelConfig,
        scales: ProductionProjectionScales,
        training: ProductionFullTrainConfig,
    ) -> Result<Self, &'static str> {
        ProductionNumericContract::derive(config, scales)?;
        let probability_bits = training.probability_gradient_fractional_bits;
        if !(15..=31).contains(&probability_bits) {
            return Err("production probability-gradient precision must be in 15..=31");
        }
        let precision_delta = probability_bits - 15;
        let output_backward_base_shift = training
            .output_backward_shift
            .unwrap_or(scales.output_shift);
        let output_backward_effective_shift = output_backward_base_shift
            .checked_add(precision_delta)
            .ok_or("production output backward shift overflow")?;
        let update_shifts = super::training::effective_learning_rate_shifts(training);
        let output_update_shift = update_shifts[11]
            .checked_add(precision_delta)
            .ok_or("production output update shift overflow")?;
        let bias_update_shift = update_shifts[12]
            .checked_add(precision_delta)
            .ok_or("production bias update shift overflow")?;
        if output_backward_effective_shift > MAX_RIGHT_SHIFT
            || output_update_shift > MAX_RIGHT_SHIFT
            || bias_update_shift > MAX_RIGHT_SHIFT
            || update_shifts[..11]
                .iter()
                .any(|&shift| shift > MAX_RIGHT_SHIFT)
        {
            return Err("production training shift exceeds numeric primitive limit");
        }

        let probability_bits = i16::from(probability_bits);
        let output_weight_bits = i16::from(scales.output_shift)
            - (PRODUCTION_ACTIVATION_FRACTIONAL_BITS - PRODUCTION_LOGIT_FRACTIONAL_BITS);
        let output_accumulator_bits = probability_bits + output_weight_bits;
        let output_arithmetic_bits =
            output_accumulator_bits - i16::from(output_backward_effective_shift);
        let mut backward_edges = vec![ProductionBackwardEdgeContract {
            name: "output_to_features",
            input_gradient_fractional_bits: probability_bits,
            weight_fractional_bits: output_weight_bits,
            accumulator_fractional_bits: output_accumulator_bits,
            right_shift: output_backward_effective_shift,
            arithmetic_output_fractional_bits: output_arithmetic_bits,
            interpreted_output_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            interpretation_delta_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS
                - output_arithmetic_bits,
            rounding: ProductionRoundingRule::RoundHalfUp,
            straight_through_nonzero_rescue: true,
        }];
        for (name, weight_bits, shift) in [
            (
                "down_to_gated",
                i16::from(scales.down_shift),
                scales.down_shift,
            ),
            (
                "up_to_mlp_input",
                i16::from(scales.up_shift),
                scales.up_shift,
            ),
            (
                "gate_to_mlp_input",
                i16::from(scales.gate_shift),
                scales.gate_shift,
            ),
            ("o_to_context", i16::from(scales.o_shift), scales.o_shift),
            (
                "q_to_attention_input",
                i16::from(scales.qkv_shift),
                scales.qkv_shift,
            ),
            (
                "k_to_attention_input",
                i16::from(scales.qkv_shift),
                scales.qkv_shift,
            ),
            (
                "v_to_attention_input",
                i16::from(scales.qkv_shift),
                scales.qkv_shift,
            ),
        ] {
            let accumulator_bits = PRODUCTION_ACTIVATION_FRACTIONAL_BITS + weight_bits;
            let arithmetic_output_bits = accumulator_bits - i16::from(shift);
            backward_edges.push(ProductionBackwardEdgeContract {
                name,
                input_gradient_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
                weight_fractional_bits: weight_bits,
                accumulator_fractional_bits: accumulator_bits,
                right_shift: shift,
                arithmetic_output_fractional_bits: arithmetic_output_bits,
                interpreted_output_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
                interpretation_delta_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS
                    - arithmetic_output_bits,
                rounding: ProductionRoundingRule::RoundHalfUp,
                straight_through_nonzero_rescue: true,
            });
        }

        let parameter_bits = [
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            i16::from(scales.qkv_shift),
            i16::from(scales.qkv_shift),
            i16::from(scales.qkv_shift),
            i16::from(scales.o_shift),
            i16::from(scales.up_shift),
            i16::from(scales.gate_shift),
            i16::from(scales.down_shift),
            output_weight_bits,
            PRODUCTION_LOGIT_FRACTIONAL_BITS,
        ];
        let raw_gradient_bits = [
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            probability_bits + PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            probability_bits,
        ];
        let mut effective_update_shifts = update_shifts;
        effective_update_shifts[11] = output_update_shift;
        effective_update_shifts[12] = bias_update_shift;
        let parameter_updates = TRAINABLE_GROUP_NAMES
            .iter()
            .enumerate()
            .map(|(index, &name)| ProductionParameterUpdateContract {
                name,
                parameter_fractional_bits: parameter_bits[index],
                raw_gradient_fractional_bits: raw_gradient_bits[index],
                update_right_shift: effective_update_shifts[index],
                real_learning_rate_power_of_two: raw_gradient_bits[index]
                    - i16::from(effective_update_shifts[index])
                    - parameter_bits[index],
                rounding: ProductionRoundingRule::RoundHalfUp,
                residual_error_feedback: true,
            })
            .collect();

        Ok(Self {
            profile: config.profile_id().unwrap_or("custom"),
            probability_gradient_fractional_bits: training.probability_gradient_fractional_bits,
            internal_gradient_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            output_backward_base_shift,
            output_backward_effective_shift,
            attention_context_gradient_left_shift: 8,
            backward_edges,
            parameter_updates,
        })
    }

    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_training_numeric_contract.v1\",",
                "\"profile\":\"{}\",\"probability_gradient_fractional_bits\":{},",
                "\"internal_gradient_fractional_bits\":{},",
                "\"output_backward_base_shift\":{},\"output_backward_effective_shift\":{},",
                "\"attention_context_gradient_left_shift\":{},\"backward_edges\":["
            ),
            self.profile,
            self.probability_gradient_fractional_bits,
            self.internal_gradient_fractional_bits,
            self.output_backward_base_shift,
            self.output_backward_effective_shift,
            self.attention_context_gradient_left_shift,
        )
        .expect("writing training numeric contract JSON to String cannot fail");
        for (index, edge) in self.backward_edges.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(
                json,
                concat!(
                    "{{\"name\":\"{}\",\"input_gradient_fractional_bits\":{},",
                    "\"weight_fractional_bits\":{},\"accumulator_fractional_bits\":{},",
                    "\"right_shift\":{},\"arithmetic_output_fractional_bits\":{},",
                    "\"interpreted_output_fractional_bits\":{},",
                    "\"interpretation_delta_bits\":{},\"rounding\":\"{}\",",
                    "\"straight_through_nonzero_rescue\":{}}}"
                ),
                edge.name,
                edge.input_gradient_fractional_bits,
                edge.weight_fractional_bits,
                edge.accumulator_fractional_bits,
                edge.right_shift,
                edge.arithmetic_output_fractional_bits,
                edge.interpreted_output_fractional_bits,
                edge.interpretation_delta_bits,
                edge.rounding.as_str(),
                edge.straight_through_nonzero_rescue,
            )
            .expect("writing training numeric contract JSON to String cannot fail");
        }
        json.push_str("],\"parameter_updates\":[");
        for (index, update) in self.parameter_updates.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(
                json,
                concat!(
                    "{{\"name\":\"{}\",\"parameter_fractional_bits\":{},",
                    "\"raw_gradient_fractional_bits\":{},\"update_right_shift\":{},",
                    "\"real_learning_rate_power_of_two\":{},\"rounding\":\"{}\",",
                    "\"residual_error_feedback\":{}}}"
                ),
                update.name,
                update.parameter_fractional_bits,
                update.raw_gradient_fractional_bits,
                update.update_right_shift,
                update.real_learning_rate_power_of_two,
                update.rounding.as_str(),
                update.residual_error_feedback,
            )
            .expect("writing training numeric contract JSON to String cannot fail");
        }
        json.push_str("]}\n");
        json
    }
}

impl ProductionNumericContract {
    pub fn derive(
        config: ProductionModelConfig,
        scales: ProductionProjectionScales,
    ) -> Result<Self, &'static str> {
        validate_config_numeric_bounds(config)?;
        if scale_shifts(scales)
            .iter()
            .any(|&shift| shift > MAX_RIGHT_SHIFT)
        {
            return Err("production projection shift exceeds numeric primitive limit");
        }

        let projections = vec![
            i8_projection("q", config.d_model, scales.qkv_shift)?,
            i8_projection("k", config.d_model, scales.qkv_shift)?,
            i8_projection("v", config.d_model, scales.qkv_shift)?,
            i8_projection("o", config.d_model, scales.o_shift)?,
            i8_projection("up", config.d_model, scales.up_shift)?,
            i8_projection("gate", config.d_model, scales.gate_shift)?,
            i8_projection("down", config.hidden_dim, scales.down_shift)?,
            output_projection(config.d_model, scales.output_shift)?,
        ];
        if projections
            .iter()
            .any(|projection| !projection.is_dimensionally_consistent())
        {
            return Err("production projection scale equation is inconsistent");
        }

        let attention = attention_bounds(config)?;
        let maximum_rms_sum_squares = checked_u64(
            (config.d_model as u128)
                .checked_mul(MAX_I16_MAGNITUDE * MAX_I16_MAGNITUDE)
                .ok_or("production RMS accumulator bound overflow")?,
            "production RMS accumulator exceeds u64",
        )?;
        let maximum_softmax_weight_sum = checked_u64(
            (config.vocab_size as u128)
                .checked_mul(i16::MAX as u128)
                .ok_or("production softmax sum bound overflow")?,
            "production softmax sum exceeds u64",
        )?;

        Ok(Self {
            profile: config.profile_id().unwrap_or("custom"),
            activation_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
            logit_fractional_bits: PRODUCTION_LOGIT_FRACTIONAL_BITS,
            rms_square_fractional_bits: PRODUCTION_RMS_SQUARE_FRACTIONAL_BITS,
            rms_epsilon_stored_units: 1,
            projections,
            attention,
            maximum_rms_sum_squares,
            maximum_softmax_weight_sum,
        })
    }

    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_numeric_contract.v1\",",
                "\"profile\":\"{}\",",
                "\"representations\":{{\"activation_fractional_bits\":{},",
                "\"logit_fractional_bits\":{},\"rms_square_fractional_bits\":{},",
                "\"rms_epsilon_stored_units\":{}}},\"projections\":["
            ),
            self.profile,
            self.activation_fractional_bits,
            self.logit_fractional_bits,
            self.rms_square_fractional_bits,
            self.rms_epsilon_stored_units,
        )
        .expect("writing numeric contract JSON to String cannot fail");
        for (index, projection) in self.projections.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(
                json,
                concat!(
                    "{{\"name\":\"{}\",\"input_fractional_bits\":{},",
                    "\"weight_fractional_bits\":{},\"accumulator_fractional_bits\":{},",
                    "\"output_fractional_bits\":{},\"right_shift\":{},",
                    "\"rounding\":\"{}\",\"maximum_absolute_accumulator\":{},",
                    "\"accumulator_bits_required\":{}}}"
                ),
                projection.name,
                projection.input_fractional_bits,
                projection.weight_fractional_bits,
                projection.accumulator_fractional_bits,
                projection.output_fractional_bits,
                projection.right_shift,
                projection.rounding.as_str(),
                projection.maximum_absolute_accumulator,
                projection.accumulator_bits_required,
            )
            .expect("writing numeric contract JSON to String cannot fail");
        }
        write!(
            json,
            concat!(
                "],\"linear_attention_bounds\":{{\"maximum_key_sum\":{},",
                "\"maximum_absolute_state_value\":{},\"maximum_denominator\":{},",
                "\"maximum_absolute_numerator\":{},\"division_rounding\":\"{}\"}},",
                "\"other_accumulator_bounds\":{{\"maximum_rms_sum_squares\":{},",
                "\"maximum_softmax_weight_sum\":{}}}}}\n"
            ),
            self.attention.maximum_key_sum,
            self.attention.maximum_absolute_state_value,
            self.attention.maximum_denominator,
            self.attention.maximum_absolute_numerator,
            self.attention.division_rounding.as_str(),
            self.maximum_rms_sum_squares,
            self.maximum_softmax_weight_sum,
        )
        .expect("writing numeric contract JSON to String cannot fail");
        json
    }
}

pub(super) fn validate_config_numeric_bounds(
    config: ProductionModelConfig,
) -> Result<(), &'static str> {
    if config.d_model == 0
        || config.heads == 0
        || !config.d_model.is_multiple_of(config.heads)
        || config.hidden_dim == 0
        || config.context_tokens == 0
        || config.vocab_size == 0
    {
        return Err("production dimensions are invalid");
    }

    let i8_i16_term = MAX_I8_MAGNITUDE * MAX_I16_MAGNITUDE;
    for dimension in [config.d_model, config.hidden_dim] {
        checked_i64_bound(
            (dimension as u128)
                .checked_mul(i8_i16_term)
                .ok_or("production projection bound overflow")?,
            "production projection accumulator exceeds i64",
        )?;
    }
    checked_i64_bound(
        (config.d_model as u128)
            .checked_mul(MAX_I16_MAGNITUDE * MAX_I16_MAGNITUDE)
            .ok_or("production output projection bound overflow")?,
        "production output accumulator exceeds i64",
    )?;
    attention_bounds(config)?;
    Ok(())
}

fn i8_projection(
    name: &'static str,
    input_dim: usize,
    right_shift: u8,
) -> Result<ProductionProjectionContract, &'static str> {
    let maximum = checked_i64_bound(
        (input_dim as u128)
            .checked_mul(MAX_I8_MAGNITUDE * MAX_I16_MAGNITUDE)
            .ok_or("production projection bound overflow")?,
        "production projection accumulator exceeds i64",
    )?;
    let weight_fractional_bits = i16::from(right_shift);
    Ok(ProductionProjectionContract {
        name,
        input_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
        weight_fractional_bits,
        accumulator_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS + weight_fractional_bits,
        output_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
        right_shift,
        rounding: ProductionRoundingRule::RoundHalfUp,
        maximum_absolute_accumulator: maximum,
        accumulator_bits_required: signed_bits_required(maximum),
    })
}

fn output_projection(
    input_dim: usize,
    right_shift: u8,
) -> Result<ProductionProjectionContract, &'static str> {
    let maximum = checked_i64_bound(
        (input_dim as u128)
            .checked_mul(MAX_I16_MAGNITUDE * MAX_I16_MAGNITUDE)
            .ok_or("production output projection bound overflow")?,
        "production output accumulator exceeds i64",
    )?;
    let weight_fractional_bits = i16::from(right_shift)
        - (PRODUCTION_ACTIVATION_FRACTIONAL_BITS - PRODUCTION_LOGIT_FRACTIONAL_BITS);
    Ok(ProductionProjectionContract {
        name: "output",
        input_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS,
        weight_fractional_bits,
        accumulator_fractional_bits: PRODUCTION_ACTIVATION_FRACTIONAL_BITS + weight_fractional_bits,
        output_fractional_bits: PRODUCTION_LOGIT_FRACTIONAL_BITS,
        right_shift,
        rounding: ProductionRoundingRule::ArithmeticRightShift,
        maximum_absolute_accumulator: maximum,
        accumulator_bits_required: signed_bits_required(maximum),
    })
}

fn attention_bounds(
    config: ProductionModelConfig,
) -> Result<ProductionAttentionBounds, &'static str> {
    let context = config.context_tokens as u128;
    let head_dim = (config.d_model / config.heads) as u128;
    let maximum_key_sum = context
        .checked_mul(LINEAR_ATTENTION_PHI_MAX)
        .ok_or("linear-attention key-sum bound overflow")?;
    let maximum_absolute_state_value = maximum_key_sum
        .checked_mul(MAX_I16_MAGNITUDE)
        .ok_or("linear-attention state bound overflow")?;
    let maximum_denominator = head_dim
        .checked_mul(LINEAR_ATTENTION_PHI_MAX)
        .and_then(|value| value.checked_mul(maximum_key_sum))
        .ok_or("linear-attention denominator bound overflow")?;
    let maximum_absolute_numerator = head_dim
        .checked_mul(LINEAR_ATTENTION_PHI_MAX)
        .and_then(|value| value.checked_mul(maximum_absolute_state_value))
        .ok_or("linear-attention numerator bound overflow")?;
    Ok(ProductionAttentionBounds {
        maximum_key_sum: checked_i64_bound(
            maximum_key_sum,
            "linear-attention key sum exceeds i64",
        )?,
        maximum_absolute_state_value: checked_i64_bound(
            maximum_absolute_state_value,
            "linear-attention state exceeds i64",
        )?,
        maximum_denominator: checked_i64_bound(
            maximum_denominator,
            "linear-attention denominator exceeds i64",
        )?,
        maximum_absolute_numerator: checked_i64_bound(
            maximum_absolute_numerator,
            "linear-attention numerator exceeds i64",
        )?,
        division_rounding: ProductionRoundingRule::NearestTiesAwayFromZero,
    })
}

fn checked_i64_bound(value: u128, error: &'static str) -> Result<u64, &'static str> {
    if value > i64::MAX as u128 {
        return Err(error);
    }
    Ok(value as u64)
}

fn checked_u64(value: u128, error: &'static str) -> Result<u64, &'static str> {
    u64::try_from(value).map_err(|_| error)
}

fn signed_bits_required(maximum_absolute_value: u64) -> u8 {
    if maximum_absolute_value == 0 {
        1
    } else {
        (u64::BITS - maximum_absolute_value.leading_zeros() + 1) as u8
    }
}

fn scale_shifts(scales: ProductionProjectionScales) -> [u8; 6] {
    [
        scales.qkv_shift,
        scales.o_shift,
        scales.up_shift,
        scales.gate_shift,
        scales.down_shift,
        scales.output_shift,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_advertised_profile_has_a_valid_numeric_contract() {
        for profile in ["p10m", "p20m", "p30m"] {
            let config = ProductionModelConfig::profile(profile).expect("advertised profile");
            let contract =
                ProductionNumericContract::derive(config, ProductionProjectionScales::default())
                    .expect("valid numeric contract");
            assert_eq!(contract.profile, profile);
            assert!(
                contract
                    .projections
                    .iter()
                    .all(ProductionProjectionContract::is_dimensionally_consistent)
            );
            assert!(contract.attention.maximum_absolute_numerator <= i64::MAX as u64);
        }
    }

    #[test]
    fn output_weight_exponent_tracks_forward_shift() {
        let config = ProductionModelConfig::profile("p10m").expect("p10m");
        let scales = ProductionProjectionScales {
            output_shift: 14,
            ..ProductionProjectionScales::default()
        };
        let contract = ProductionNumericContract::derive(config, scales).expect("contract");
        let output = contract
            .projections
            .iter()
            .find(|projection| projection.name == "output")
            .expect("output projection");
        assert_eq!(output.weight_fractional_bits, 7);
        assert_eq!(
            output.output_fractional_bits,
            PRODUCTION_LOGIT_FRACTIONAL_BITS
        );
        assert_eq!(
            output.rounding,
            ProductionRoundingRule::ArithmeticRightShift
        );
    }

    #[test]
    fn numeric_contract_json_exposes_scales_rounding_and_bounds() {
        let config = ProductionModelConfig::profile("p20m").expect("p20m");
        let json = ProductionNumericContract::derive(config, ProductionProjectionScales::default())
            .expect("contract")
            .to_json_line();
        assert!(json.contains("nsrl.production_numeric_contract.v1"));
        assert!(json.contains("arithmetic_right_shift_toward_negative_infinity"));
        assert!(json.contains("maximum_absolute_numerator"));
    }

    #[test]
    fn training_contract_exposes_output_damping_and_real_learning_rates() {
        let config = ProductionModelConfig::profile("p10m").expect("p10m");
        let scales = ProductionProjectionScales::default();
        let training = ProductionFullTrainConfig::default();
        let contract = ProductionTrainingNumericContract::derive(config, scales, training)
            .expect("training contract");
        let output_backward = contract
            .backward_edges
            .iter()
            .find(|edge| edge.name == "output_to_features")
            .expect("output backward edge");
        assert_eq!(output_backward.interpretation_delta_bits, 7);
        assert!(
            contract
                .backward_edges
                .iter()
                .filter(|edge| edge.name != "output_to_features")
                .all(|edge| edge.interpretation_delta_bits == 0)
        );
        let output_update = contract
            .parameter_updates
            .iter()
            .find(|update| update.name == "output")
            .expect("output update");
        assert_eq!(output_update.raw_gradient_fractional_bits, 30);
        assert_eq!(output_update.parameter_fractional_bits, 1);
        assert_eq!(output_update.real_learning_rate_power_of_two, 5);
        assert!(
            contract
                .to_json_line()
                .contains("attention_context_gradient_left_shift")
        );
    }
}
