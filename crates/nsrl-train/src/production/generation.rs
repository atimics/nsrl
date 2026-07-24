//! Deterministic autoregressive generation for the production model.
//!
//! # Design
//!
//! Generation uses the linear-attention incremental cache so each new
//! token only computes one step of attention rather than full-window
//! replay.  The cache state (`LinearAttentionState`) is updated in place
//! and the output head produces logits for the next token.
//!
//! # Configuration
//!
//! [`ProductionGenerationConfig`] bundles sampling strategy, repeat
//! penalties, and stopping criteria.  The config is versioned so
//! generation traces are byte-replayable.

use std::fmt::Write;

use nsrl_core::{
    GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionState, LinearAttentionStepWorkspace,
    SelfAttentionI16Params, base2_softmax_i32_q15, gated_mlp_i16_q15_checked,
    linear_attention_state_lengths, linear_attention_step_i16_q15_checked,
    rms_norm_i16_q15_checked,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID};

use super::{
    PRODUCTION_RMS_EPSILON, ProductionForward, ProductionModelV1, add_residual, linear_params,
    output_logits, scales,
};
use crate::TrainError;

pub const PRODUCTION_GENERATION_SCHEMA: &str = "nsrl.production_generation.v2";

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
    pub cache_state_bytes: usize,
    pub cache_workspace_bytes: usize,
    pub cache_tokens_processed: usize,
    pub steps_beyond_training_context: usize,
    pub prefill_residual_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub steps: Vec<ProductionGenerationStepTrace>,
}

impl ProductionGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"{}\",\"execution\":\"incremental_linear_attention_cache_v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\"}},",
                "\"config\":{{\"strategy\":\"{}\",\"context_tokens\":{},",
                "\"max_new_tokens\":{},\"top_k\":{},\"sample_seed\":{},",
                "\"stop_on_eos\":{}}},",
                "\"prompt\":{{\"token_count\":{},\"token_hash\":\"0x{:016x}\"}},",
                "\"generation\":{{\"token_count\":{},\"token_hash\":\"0x{:016x}\",",
                "\"stopped_on_eos\":{},\"steps_beyond_training_context\":{},",
                "\"prefill_residual_saturation_count\":{},",
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
            self.steps_beyond_training_context,
            self.prefill_residual_saturation_count,
            self.residual_saturation_count,
        );
        for (index, token) in self.generated_tokens.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(output, "{token}").expect("writing JSON to String cannot fail");
        }
        write!(
            output,
            concat!(
                "],\"cache\":{{\"state_bytes\":{},\"workspace_bytes\":{},",
                "\"tokens_processed\":{}}},\"steps\":["
            ),
            self.cache_state_bytes, self.cache_workspace_bytes, self.cache_tokens_processed,
        )
        .expect("writing JSON to String cannot fail");
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

#[derive(Debug)]
struct ProductionDecoderLayerState {
    state_kv: Vec<i64>,
    key_sums: Vec<i64>,
}

#[derive(Debug)]
struct ProductionDecoderWorkspace {
    hidden: Vec<i16>,
    attention_input: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_input: Vec<i16>,
    mlp_output: Vec<i16>,
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    attention_context: Vec<i16>,
    up: Vec<i16>,
    gate: Vec<i16>,
    gated: Vec<i16>,
    features: Vec<i16>,
}

/// Exact streaming decoder for the causal linear-attention production model.
///
/// Each token performs one row of every layer and retains only the per-layer
/// linear-attention sufficient statistics. Prefixes no longer need to be
/// replayed for each generated token.
#[derive(Debug)]
pub struct ProductionDecoder<'model> {
    model: &'model ProductionModelV1,
    layers: Vec<ProductionDecoderLayerState>,
    workspace: ProductionDecoderWorkspace,
    qkv_scales: Vec<nsrl_core::FixedScale>,
    o_scales: Vec<nsrl_core::FixedScale>,
    up_scales: Vec<nsrl_core::FixedScale>,
    gate_scales: Vec<nsrl_core::FixedScale>,
    down_scales: Vec<nsrl_core::FixedScale>,
    tokens_processed: usize,
}

impl<'model> ProductionDecoder<'model> {
    pub fn new(model: &'model ProductionModelV1) -> Result<Self, TrainError> {
        model.validate()?;
        let config = model.config;
        let (state_len, key_sum_len) = linear_attention_state_lengths(config.d_model, config.heads)
            .ok_or(TrainError::InvalidConfig)?;
        let layers = (0..config.layers)
            .map(|_| ProductionDecoderLayerState {
                state_kv: vec![0; state_len],
                key_sums: vec![0; key_sum_len],
            })
            .collect();
        Ok(Self {
            model,
            layers,
            workspace: ProductionDecoderWorkspace {
                hidden: vec![0; config.d_model],
                attention_input: vec![0; config.d_model],
                attention_output: vec![0; config.d_model],
                attention_residual: vec![0; config.d_model],
                mlp_input: vec![0; config.d_model],
                mlp_output: vec![0; config.d_model],
                q: vec![0; config.d_model],
                k: vec![0; config.d_model],
                v: vec![0; config.d_model],
                attention_context: vec![0; config.d_model],
                up: vec![0; config.hidden_dim],
                gate: vec![0; config.hidden_dim],
                gated: vec![0; config.hidden_dim],
                features: vec![0; config.d_model],
            },
            qkv_scales: scales(config.d_model, model.scales.qkv_shift),
            o_scales: scales(config.d_model, model.scales.o_shift),
            up_scales: scales(config.hidden_dim, model.scales.up_shift),
            gate_scales: scales(config.hidden_dim, model.scales.gate_shift),
            down_scales: scales(config.d_model, model.scales.down_shift),
            tokens_processed: 0,
        })
    }

    pub const fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    pub fn state_bytes(&self) -> usize {
        self.layers.iter().fold(0, |total, layer| {
            total
                + core::mem::size_of_val(layer.state_kv.as_slice())
                + core::mem::size_of_val(layer.key_sums.as_slice())
        })
    }

    pub fn workspace_bytes(&self) -> usize {
        let workspace = &self.workspace;
        [
            workspace.hidden.len(),
            workspace.attention_input.len(),
            workspace.attention_output.len(),
            workspace.attention_residual.len(),
            workspace.mlp_input.len(),
            workspace.mlp_output.len(),
            workspace.q.len(),
            workspace.k.len(),
            workspace.v.len(),
            workspace.attention_context.len(),
            workspace.up.len(),
            workspace.gate.len(),
            workspace.gated.len(),
            workspace.features.len(),
        ]
        .into_iter()
        .sum::<usize>()
            * core::mem::size_of::<i16>()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.state_kv.fill(0);
            layer.key_sums.fill(0);
        }
        self.tokens_processed = 0;
    }

    pub fn step(&mut self, token: u32) -> Result<ProductionForward, TrainError> {
        let model = self.model;
        let config = model.config;
        if token as usize >= config.vocab_size {
            return Err(TrainError::InvalidConfig);
        }
        let embedding_start = token as usize * config.d_model;
        self.workspace
            .hidden
            .copy_from_slice(&model.embeddings[embedding_start..embedding_start + config.d_model]);

        let matrix = config.d_model * config.d_model;
        let up_matrix = config.d_model * config.hidden_dim;
        let down_matrix = config.hidden_dim * config.d_model;
        let mut residual_saturation_count = 0_usize;
        for layer in 0..config.layers {
            let rms = layer * config.d_model..(layer + 1) * config.d_model;
            rms_norm_i16_q15_checked(
                &self.workspace.hidden,
                &model.attention_rms_weights[rms.clone()],
                PRODUCTION_RMS_EPSILON,
                &mut self.workspace.attention_input,
            )
            .ok_or(TrainError::CoreRejected(
                "production_incremental_attention_rms",
            ))?;
            let range = layer * matrix..(layer + 1) * matrix;
            let params = SelfAttentionI16Params {
                q: linear_params(
                    &model.q_weights[range.clone()],
                    &self.qkv_scales,
                    config.d_model,
                    config.d_model,
                ),
                k: linear_params(
                    &model.k_weights[range.clone()],
                    &self.qkv_scales,
                    config.d_model,
                    config.d_model,
                ),
                v: linear_params(
                    &model.v_weights[range.clone()],
                    &self.qkv_scales,
                    config.d_model,
                    config.d_model,
                ),
                o: linear_params(
                    &model.o_weights[range],
                    &self.o_scales,
                    config.d_model,
                    config.d_model,
                ),
                seq_len: self.tokens_processed.saturating_add(1),
                d_model: config.d_model,
                heads: config.heads,
                causal: true,
            };
            let layer_state = &mut self.layers[layer];
            linear_attention_step_i16_q15_checked(
                &self.workspace.attention_input,
                params,
                LinearAttentionStepWorkspace {
                    q: &mut self.workspace.q,
                    k: &mut self.workspace.k,
                    v: &mut self.workspace.v,
                    context: &mut self.workspace.attention_context,
                },
                LinearAttentionState {
                    state_kv: &mut layer_state.state_kv,
                    key_sums: &mut layer_state.key_sums,
                },
                &mut self.workspace.attention_output,
            )
            .ok_or(TrainError::CoreRejected(
                "production_incremental_linear_attention",
            ))?;
            residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
                &self.workspace.hidden,
                &self.workspace.attention_output,
                &mut self.workspace.attention_residual,
            ));
            rms_norm_i16_q15_checked(
                &self.workspace.attention_residual,
                &model.mlp_rms_weights[rms],
                PRODUCTION_RMS_EPSILON,
                &mut self.workspace.mlp_input,
            )
            .ok_or(TrainError::CoreRejected("production_incremental_mlp_rms"))?;
            let up_range = layer * up_matrix..(layer + 1) * up_matrix;
            let down_range = layer * down_matrix..(layer + 1) * down_matrix;
            gated_mlp_i16_q15_checked(
                &self.workspace.mlp_input,
                GatedMlpI16Params {
                    up: linear_params(
                        &model.up_weights[up_range.clone()],
                        &self.up_scales,
                        config.d_model,
                        config.hidden_dim,
                    ),
                    gate: linear_params(
                        &model.gate_weights[up_range],
                        &self.gate_scales,
                        config.d_model,
                        config.hidden_dim,
                    ),
                    down: linear_params(
                        &model.down_weights[down_range],
                        &self.down_scales,
                        config.hidden_dim,
                        config.d_model,
                    ),
                    seq_len: 1,
                    d_model: config.d_model,
                    hidden_dim: config.hidden_dim,
                },
                GatedMlpWorkspace {
                    up: &mut self.workspace.up,
                    gate: &mut self.workspace.gate,
                    gated: &mut self.workspace.gated,
                },
                &mut self.workspace.mlp_output,
            )
            .ok_or(TrainError::CoreRejected("production_incremental_gated_mlp"))?;
            residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
                &self.workspace.attention_residual,
                &self.workspace.mlp_output,
                &mut self.workspace.hidden,
            ));
        }

        rms_norm_i16_q15_checked(
            &self.workspace.hidden,
            &model.final_rms_weights,
            PRODUCTION_RMS_EPSILON,
            &mut self.workspace.features,
        )
        .ok_or(TrainError::CoreRejected("production_incremental_final_rms"))?;
        let logits_q8 = output_logits(model, &self.workspace.features)?;
        let mut probabilities_q15 = vec![0_i16; config.vocab_size];
        base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
            .ok_or(TrainError::CoreRejected("production_incremental_softmax"))?;
        self.tokens_processed = self.tokens_processed.saturating_add(1);
        Ok(ProductionForward {
            features_q15: self.workspace.features.clone(),
            logits_q8,
            probabilities_q15,
            residual_saturation_count,
        })
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
        || prompt_tokens.len() > config.context_tokens
        || config.max_new_tokens == 0
        || config.top_k == 0
        || config.top_k > model.config.vocab_size.saturating_sub(2)
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut stopped_on_eos = false;
    let mut steps_beyond_training_context = 0_usize;
    let mut decoder = ProductionDecoder::new(model)?;
    let cache_state_bytes = decoder.state_bytes();
    let cache_workspace_bytes = decoder.workspace_bytes();
    let mut current_forward = None;
    let mut prefill_residual_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;
    for &token in prompt_tokens {
        let forward = decoder.step(token)?;
        prefill_residual_saturation_count =
            prefill_residual_saturation_count.saturating_add(forward.residual_saturation_count);
        residual_saturation_count =
            residual_saturation_count.saturating_add(forward.residual_saturation_count);
        current_forward = Some(forward);
    }

    for step_index in 0..config.max_new_tokens {
        let forward = current_forward
            .as_ref()
            .ok_or(TrainError::CoreRejected("production_generation_prefill"))?;
        let context_len = decoder.tokens_processed();
        steps_beyond_training_context = steps_beyond_training_context
            .saturating_add(usize::from(context_len > model.config.context_tokens));
        let selection = select_token(
            &forward.logits_q8,
            &forward.probabilities_q15,
            config.top_k,
            config.sample_seed,
            step_index,
        )?;
        let selected_token = selection.token as u32;
        generated_tokens.push(selected_token);
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
        if step_index + 1 < config.max_new_tokens {
            let forward = decoder.step(selected_token)?;
            residual_saturation_count =
                residual_saturation_count.saturating_add(forward.residual_saturation_count);
            current_forward = Some(forward);
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
        cache_state_bytes,
        cache_workspace_bytes,
        cache_tokens_processed: decoder.tokens_processed(),
        steps_beyond_training_context,
        prefill_residual_saturation_count,
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
    fn incremental_decoder_matches_full_forward_on_every_training_length_prefix() {
        let model = model();
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut decoder = ProductionDecoder::new(&model).expect("decoder");
        for prefix_len in 1..=tokens.len() {
            let incremental = decoder.step(tokens[prefix_len - 1]).expect("incremental");
            let full = super::super::forward_production_model(&model, &tokens[..prefix_len])
                .expect("full");
            assert_eq!(incremental, full, "prefix length {prefix_len}");
        }
        assert_eq!(decoder.tokens_processed(), tokens.len());
        assert!(decoder.state_bytes() > 0);
        assert!(decoder.workspace_bytes() > 0);
        decoder.reset();
        assert_eq!(decoder.tokens_processed(), 0);
        assert_eq!(
            decoder.step(tokens[0]).expect("replay"),
            super::super::forward_production_model(&model, &tokens[..1]).expect("full replay")
        );
    }

    #[test]
    fn greedy_generation_masks_nonsemantic_special_tokens_and_streams_beyond_context() {
        let mut model = model();
        model.output_bias_q8[BOS_TOKEN_ID as usize] = 30_000;
        model.output_bias_q8[PAD_TOKEN_ID as usize] = 29_000;
        model.output_bias_q8[42] = 28_000;
        let trace = generate_production_model(
            &model,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            ProductionGenerationConfig::greedy(8, 3),
        )
        .expect("generation");
        assert_eq!(trace.generated_tokens, [42, 42, 42]);
        assert_eq!(trace.steps_beyond_training_context, 2);
        assert_eq!(
            trace
                .steps
                .iter()
                .map(|step| step.context_tokens)
                .collect::<Vec<_>>(),
            [8, 9, 10]
        );
        assert_eq!(trace.cache_tokens_processed, 10);
        let json = trace.to_json_line();
        assert!(json.contains(PRODUCTION_GENERATION_SCHEMA));
        assert!(json.contains("incremental_linear_attention_cache_v1"));
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
        assert!(
            generate_production_model(
                &model,
                &[1, 2, 3, 4, 5],
                ProductionGenerationConfig::greedy(4, 1),
            )
            .is_err()
        );
    }
}
