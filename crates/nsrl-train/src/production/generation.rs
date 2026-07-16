use std::fmt::Write;

use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID};

use super::{ProductionModelV1, forward_production_model};
use crate::TrainError;

pub const PRODUCTION_GENERATION_SCHEMA: &str = "nsrl.production_generation.v1";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGenerationConfig {
    pub context_tokens: usize,
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub sample_seed: u64,
    pub stop_on_eos: bool,
}

impl ProductionGenerationConfig {
    pub const fn greedy(context_tokens: usize, max_new_tokens: usize) -> Self {
        Self {
            context_tokens,
            max_new_tokens,
            top_k: 1,
            sample_seed: 0,
            stop_on_eos: true,
        }
    }

    pub const fn deterministic_top_k(
        context_tokens: usize,
        max_new_tokens: usize,
        top_k: usize,
        sample_seed: u64,
    ) -> Self {
        Self {
            context_tokens,
            max_new_tokens,
            top_k,
            sample_seed,
            stop_on_eos: true,
        }
    }

    pub const fn strategy(self) -> &'static str {
        if self.top_k == 1 {
            "greedy"
        } else {
            "deterministic_top_k_probability_sample"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGenerationStepTrace {
    pub step_index: usize,
    pub context_tokens: usize,
    pub selected_token: u32,
    pub selected_logit_q8: i32,
    pub selected_probability_q15: i16,
    pub candidate_count: usize,
    pub selection_mass_q15: u64,
    pub selection_draw: u64,
    pub residual_saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGenerationTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub model_hash: u64,
    pub tokenizer_hash: u64,
    pub config: ProductionGenerationConfig,
    pub prompt_token_count: usize,
    pub prompt_token_hash: u64,
    pub generated_tokens: Vec<u32>,
    pub generated_token_hash: u64,
    pub stopped_on_eos: bool,
    pub steps_with_rolling_context: usize,
    pub residual_saturation_count: usize,
    pub steps: Vec<ProductionGenerationStepTrace>,
}

impl ProductionGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"{}\",\"execution\":\"full_window_replay_v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\"}},",
                "\"config\":{{\"strategy\":\"{}\",\"context_tokens\":{},",
                "\"max_new_tokens\":{},\"top_k\":{},\"sample_seed\":{},",
                "\"stop_on_eos\":{}}},",
                "\"prompt\":{{\"token_count\":{},\"token_hash\":\"0x{:016x}\"}},",
                "\"generation\":{{\"token_count\":{},\"token_hash\":\"0x{:016x}\",",
                "\"stopped_on_eos\":{},\"steps_with_rolling_context\":{},",
                "\"residual_saturation_count\":{},\"tokens\":["
            ),
            PRODUCTION_GENERATION_SCHEMA,
            self.profile,
            self.parameter_count,
            self.model_hash,
            self.tokenizer_hash,
            self.config.strategy(),
            self.config.context_tokens,
            self.config.max_new_tokens,
            self.config.top_k,
            self.config.sample_seed,
            self.config.stop_on_eos,
            self.prompt_token_count,
            self.prompt_token_hash,
            self.generated_tokens.len(),
            self.generated_token_hash,
            self.stopped_on_eos,
            self.steps_with_rolling_context,
            self.residual_saturation_count,
        );
        for (index, token) in self.generated_tokens.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(output, "{token}").expect("writing JSON to String cannot fail");
        }
        output.push_str("],\"steps\":[");
        for (index, step) in self.steps.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                concat!(
                    "{{\"step_index\":{},\"context_tokens\":{},\"selected_token\":{},",
                    "\"selected_logit_q8\":{},\"selected_probability_q15\":{},",
                    "\"candidate_count\":{},\"selection_mass_q15\":{},",
                    "\"selection_draw\":{},\"residual_saturation_count\":{}}}"
                ),
                step.step_index,
                step.context_tokens,
                step.selected_token,
                step.selected_logit_q8,
                step.selected_probability_q15,
                step.candidate_count,
                step.selection_mass_q15,
                step.selection_draw,
                step.residual_saturation_count,
            )
            .expect("writing JSON to String cannot fail");
        }
        output.push_str("]}}\n");
        output
    }
}

pub fn generate_production_model(
    model: &ProductionModelV1,
    prompt_tokens: &[u32],
    config: ProductionGenerationConfig,
) -> Result<ProductionGenerationTrace, TrainError> {
    model.validate()?;
    if prompt_tokens.is_empty()
        || prompt_tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
        || config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_new_tokens == 0
        || config.top_k == 0
        || config.top_k > model.config.vocab_size.saturating_sub(2)
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut history = prompt_tokens.to_vec();
    let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut stopped_on_eos = false;
    let mut steps_with_rolling_context = 0_usize;
    let mut residual_saturation_count = 0_usize;

    for step_index in 0..config.max_new_tokens {
        let context_start = history.len().saturating_sub(config.context_tokens);
        steps_with_rolling_context =
            steps_with_rolling_context.saturating_add(usize::from(context_start > 0));
        let context = &history[context_start..];
        let context_len = context.len();
        let forward = forward_production_model(model, context)?;
        let selection = select_token(
            &forward.logits_q8,
            &forward.probabilities_q15,
            config.top_k,
            config.sample_seed,
            step_index,
        )?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(forward.residual_saturation_count);
        let selected_token = selection.token as u32;
        generated_tokens.push(selected_token);
        history.push(selected_token);
        steps.push(ProductionGenerationStepTrace {
            step_index,
            context_tokens: context_len,
            selected_token,
            selected_logit_q8: forward.logits_q8[selection.token],
            selected_probability_q15: forward.probabilities_q15[selection.token],
            candidate_count: selection.candidate_count,
            selection_mass_q15: selection.mass,
            selection_draw: selection.draw,
            residual_saturation_count: forward.residual_saturation_count,
        });
        if selected_token == EOS_TOKEN_ID && config.stop_on_eos {
            stopped_on_eos = true;
            break;
        }
    }

    Ok(ProductionGenerationTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        model_hash: model.model_hash(),
        tokenizer_hash: model.tokenizer_hash,
        config,
        prompt_token_count: prompt_tokens.len(),
        prompt_token_hash: hash_tokens(prompt_tokens),
        generated_token_hash: hash_tokens(&generated_tokens),
        generated_tokens,
        stopped_on_eos,
        steps_with_rolling_context,
        residual_saturation_count,
        steps,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenSelection {
    token: usize,
    candidate_count: usize,
    mass: u64,
    draw: u64,
}

fn select_token(
    logits_q8: &[i32],
    probabilities_q15: &[i16],
    top_k: usize,
    sample_seed: u64,
    step_index: usize,
) -> Result<TokenSelection, TrainError> {
    if logits_q8.is_empty() || logits_q8.len() != probabilities_q15.len() || top_k == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let mut candidates = logits_q8
        .iter()
        .copied()
        .enumerate()
        .filter(|&(token, _)| !matches!(token as u32, BOS_TOKEN_ID | PAD_TOKEN_ID))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|(left_token, left_logit), (right_token, right_logit)| {
        right_logit
            .cmp(left_logit)
            .then_with(|| left_token.cmp(right_token))
    });
    candidates.truncate(top_k.min(candidates.len()));
    let Some(&(greedy_token, _)) = candidates.first() else {
        return Err(TrainError::InvalidConfig);
    };
    if top_k == 1 {
        return Ok(TokenSelection {
            token: greedy_token,
            candidate_count: 1,
            mass: u64::try_from(probabilities_q15[greedy_token].max(0)).unwrap_or(0),
            draw: 0,
        });
    }

    let mass = candidates.iter().fold(0_u64, |sum, (token, _)| {
        sum.saturating_add(u64::try_from(probabilities_q15[*token].max(0)).unwrap_or(0))
    });
    if mass == 0 {
        return Ok(TokenSelection {
            token: greedy_token,
            candidate_count: candidates.len(),
            mass,
            draw: 0,
        });
    }
    let draw = splitmix64(sample_seed ^ (step_index as u64).rotate_left(29)) % mass;
    let mut cursor = draw;
    for (token, _) in &candidates {
        let weight = u64::try_from(probabilities_q15[*token].max(0)).unwrap_or(0);
        if cursor < weight {
            return Ok(TokenSelection {
                token: *token,
                candidate_count: candidates.len(),
                mass,
                draw,
            });
        }
        cursor = cursor.saturating_sub(weight);
    }
    Err(TrainError::CoreRejected(
        "production_generation_sampling_mass",
    ))
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |mut hash, token| {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::production::ProductionModelConfig;

    fn model() -> ProductionModelV1 {
        let config = ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 2,
            hidden_dim: 48,
            context_tokens: 8,
        };
        ProductionModelV1::new_initial(config, 0x1234, 7).expect("model")
    }

    #[test]
    fn greedy_generation_masks_nonsemantic_special_tokens_and_rolls_context() {
        let mut model = model();
        model.output_bias_q8[BOS_TOKEN_ID as usize] = 30_000;
        model.output_bias_q8[PAD_TOKEN_ID as usize] = 29_000;
        model.output_bias_q8[42] = 28_000;
        let trace = generate_production_model(
            &model,
            &[1, 2, 3, 4, 5],
            ProductionGenerationConfig::greedy(4, 3),
        )
        .expect("generation");
        assert_eq!(trace.generated_tokens, [42, 42, 42]);
        assert_eq!(trace.steps_with_rolling_context, 3);
        assert_eq!(
            trace
                .steps
                .iter()
                .map(|step| step.context_tokens)
                .collect::<Vec<_>>(),
            [4, 4, 4]
        );
        assert!(trace.to_json_line().contains(PRODUCTION_GENERATION_SCHEMA));
    }

    #[test]
    fn generation_stops_on_eos() {
        let mut model = model();
        model.output_bias_q8[EOS_TOKEN_ID as usize] = 30_000;
        let trace = generate_production_model(
            &model,
            &[1, 2, 3, 4],
            ProductionGenerationConfig::greedy(4, 8),
        )
        .expect("generation");
        assert_eq!(trace.generated_tokens, [EOS_TOKEN_ID]);
        assert!(trace.stopped_on_eos);
        assert_eq!(trace.steps.len(), 1);
    }

    #[test]
    fn seeded_top_k_generation_replays_exactly() {
        let mut model = model();
        model.output_bias_q8[10] = 20_000;
        model.output_bias_q8[11] = 19_900;
        let config = ProductionGenerationConfig::deterministic_top_k(8, 6, 2, 19);
        let left = generate_production_model(&model, &[7, 8, 9], config).expect("left");
        let right = generate_production_model(&model, &[7, 8, 9], config).expect("right");
        assert_eq!(left, right);
        assert!(
            left.generated_tokens
                .iter()
                .all(|token| [10, 11].contains(token))
        );
        assert!(left.steps.iter().all(|step| step.candidate_count == 2));
    }

    #[test]
    fn generation_rejects_empty_prompt_and_invalid_limits() {
        let model = model();
        assert!(
            generate_production_model(&model, &[], ProductionGenerationConfig::greedy(4, 1),)
                .is_err()
        );
        assert!(
            generate_production_model(&model, &[1], ProductionGenerationConfig::greedy(9, 1),)
                .is_err()
        );
    }
}
