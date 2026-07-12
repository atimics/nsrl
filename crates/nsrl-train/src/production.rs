//! Variable-vocabulary production decoder artifact and bounded integer smoke runtime.
//!
//! This module is separate from MT5/MT6 so their byte-vocabulary artifacts and
//! frozen proof semantics remain unchanged.

use nsrl_core::{
    FixedScale, GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionWorkspace, LinearI16I8Params,
    SelfAttentionI16Params, base2_softmax_i32_q15, gated_mlp_i16_q15_checked,
    linear_attention_i16_q15_checked, rms_norm_i16_q15_checked, saturate_i16,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use crate::{PRODUCTION_MODEL_V1_MAGIC, TrainError};

mod training;
pub use training::{
    ProductionFullTrainConfig, ProductionFullTrainTrace, ProductionOptimizerStateV1,
    train_production_full_smoke,
};

pub const PRODUCTION_MODEL_V1_SCHEMA: &str = "nsrl.production_model.v1";
pub const PRODUCTION_MODEL_V1_VERSION: u32 = 1;
const PRODUCTION_RMS_EPSILON: u64 = 1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub hidden_dim: usize,
    pub context_tokens: usize,
}

impl ProductionModelConfig {
    pub fn profile(id: &str) -> Option<Self> {
        match id {
            "p10m" => Some(Self {
                vocab_size: 8_192,
                d_model: 256,
                heads: 8,
                layers: 6,
                hidden_dim: 768,
                context_tokens: 256,
            }),
            "p20m" => Some(Self {
                vocab_size: 8_192,
                d_model: 384,
                heads: 8,
                layers: 8,
                hidden_dim: 1_152,
                context_tokens: 256,
            }),
            "p30m" => Some(Self {
                vocab_size: 8_192,
                d_model: 448,
                heads: 8,
                layers: 8,
                hidden_dim: 1_344,
                context_tokens: 256,
            }),
            _ => None,
        }
    }

    pub fn profile_id(self) -> Option<&'static str> {
        ["p10m", "p20m", "p30m"]
            .into_iter()
            .find(|&id| Self::profile(id) == Some(self))
    }

    pub fn validate(self) -> Result<(), TrainError> {
        if self.vocab_size <= EOS_TOKEN_ID as usize
            || self.d_model == 0
            || self.heads == 0
            || !self.d_model.is_multiple_of(self.heads)
            || !((self.d_model / self.heads).is_power_of_two())
            || self.layers == 0
            || self.hidden_dim == 0
            || self.context_tokens == 0
            || [
                self.vocab_size,
                self.d_model,
                self.heads,
                self.layers,
                self.hidden_dim,
                self.context_tokens,
            ]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
        {
            return Err(TrainError::InvalidConfig);
        }
        self.parameter_count().ok_or(TrainError::InvalidConfig)?;
        Ok(())
    }

    pub fn parameter_count(self) -> Option<usize> {
        let embeddings_and_output = self.vocab_size.checked_mul(self.d_model)?.checked_mul(2)?;
        let attention = self.d_model.checked_mul(self.d_model)?.checked_mul(4)?;
        let mlp = self.d_model.checked_mul(self.hidden_dim)?.checked_mul(3)?;
        let rms = self.d_model.checked_mul(2)?;
        embeddings_and_output
            .checked_add(
                self.layers
                    .checked_mul(attention.checked_add(mlp)?.checked_add(rms)?)?,
            )?
            .checked_add(self.d_model)?
            .checked_add(self.vocab_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionProjectionScales {
    pub qkv_shift: u8,
    pub o_shift: u8,
    pub up_shift: u8,
    pub gate_shift: u8,
    pub down_shift: u8,
    pub output_shift: u8,
}

impl Default for ProductionProjectionScales {
    fn default() -> Self {
        Self {
            qkv_shift: 8,
            o_shift: 8,
            up_shift: 10,
            gate_shift: 10,
            down_shift: 12,
            output_shift: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionModelV1 {
    pub config: ProductionModelConfig,
    pub tokenizer_hash: u64,
    pub initialization_seed: u64,
    pub scales: ProductionProjectionScales,
    pub embeddings: Vec<i16>,
    pub attention_rms_weights: Vec<i16>,
    pub mlp_rms_weights: Vec<i16>,
    pub final_rms_weights: Vec<i16>,
    pub q_weights: Vec<i8>,
    pub k_weights: Vec<i8>,
    pub v_weights: Vec<i8>,
    pub o_weights: Vec<i8>,
    pub up_weights: Vec<i8>,
    pub gate_weights: Vec<i8>,
    pub down_weights: Vec<i8>,
    pub output_weights: Vec<i16>,
    pub output_bias_q8: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionForward {
    pub features_q15: Vec<i16>,
    pub logits_q8: Vec<i32>,
    pub probabilities_q15: Vec<i16>,
    pub residual_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionSmokeConfig {
    pub context_tokens: usize,
    pub max_windows: usize,
    pub epochs: usize,
    pub feature_shift: u8,
    pub bias_step_q8: i32,
    pub margin_q8: i32,
}

impl Default for ProductionSmokeConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4,
            max_windows: 8,
            epochs: 2,
            feature_shift: 13,
            bias_step_q8: 4,
            margin_q8: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionSmokeTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub epochs: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub updates: usize,
    pub weight_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
}

impl ProductionSmokeTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_smoke.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"scope\":\"output_head_perceptron\",\"context_tokens\":{},\"windows\":{},\"epochs\":{},\"updates\":{}}},",
                "\"evaluation\":{{\"initial_mistakes\":{},\"final_mistakes\":{}}},",
                "\"health\":{{\"weight_saturation_count\":{},\"residual_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\"}},",
                "\"known_non_claims\":[\"output_head_only_smoke_not_full_layer_training\",\"not_float_twin\",\"not_open_generation_quality\"]}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.epochs,
            self.updates,
            self.initial_mistakes,
            self.final_mistakes,
            self.weight_saturation_count,
            self.residual_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
        )
    }
}

impl ProductionModelV1 {
    pub fn new_initial(
        config: ProductionModelConfig,
        tokenizer_hash: u64,
        initialization_seed: u64,
    ) -> Result<Self, TrainError> {
        config.validate()?;
        if tokenizer_hash == 0 {
            return Err(TrainError::InvalidConfig);
        }
        let matrix = checked_product(config.d_model, config.d_model)?;
        let up = checked_product(config.d_model, config.hidden_dim)?;
        let down = checked_product(config.hidden_dim, config.d_model)?;
        let embeddings = initial_i16_tensor(
            checked_product(config.vocab_size, config.d_model)?,
            initialization_seed ^ 0x6a09_e667_f3bc_c909,
            512,
        );
        let model = Self {
            config,
            tokenizer_hash,
            initialization_seed,
            scales: ProductionProjectionScales::default(),
            embeddings,
            attention_rms_weights: vec![i16::MAX; checked_product(config.layers, config.d_model)?],
            mlp_rms_weights: vec![i16::MAX; checked_product(config.layers, config.d_model)?],
            final_rms_weights: vec![i16::MAX; config.d_model],
            q_weights: stacked_identity(config.layers, config.d_model, 16),
            k_weights: stacked_identity(config.layers, config.d_model, 16),
            v_weights: stacked_identity(config.layers, config.d_model, 16),
            o_weights: stacked_identity(config.layers, config.d_model, 8),
            up_weights: initial_i8_tensor(
                checked_product(config.layers, up)?,
                initialization_seed ^ 0x510e_527f_ade6_82d1,
                2,
            ),
            gate_weights: initial_i8_tensor(
                checked_product(config.layers, up)?,
                initialization_seed ^ 0x9b05_688c_2b3e_6c1f,
                2,
            ),
            down_weights: initial_i8_tensor(
                checked_product(config.layers, down)?,
                initialization_seed ^ 0x1f83_d9ab_fb41_bd6b,
                1,
            ),
            output_weights: vec![0_i16; checked_product(config.vocab_size, config.d_model)?],
            output_bias_q8: vec![0_i32; config.vocab_size],
        };
        debug_assert_eq!(model.q_weights.len(), config.layers * matrix);
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), TrainError> {
        self.config.validate()?;
        if self.tokenizer_hash == 0 || scale_shifts(self.scales).iter().any(|&shift| shift > 30) {
            return Err(TrainError::InvalidModel(
                "invalid production model metadata",
            ));
        }
        let config = self.config;
        let matrix = checked_product(config.d_model, config.d_model)?;
        let rms = checked_product(config.layers, config.d_model)?;
        let up = checked_product(
            config.layers,
            checked_product(config.d_model, config.hidden_dim)?,
        )?;
        let down = checked_product(
            config.layers,
            checked_product(config.hidden_dim, config.d_model)?,
        )?;
        if self.embeddings.len() != checked_product(config.vocab_size, config.d_model)?
            || self.attention_rms_weights.len() != rms
            || self.mlp_rms_weights.len() != rms
            || self.final_rms_weights.len() != config.d_model
            || self.q_weights.len() != checked_product(config.layers, matrix)?
            || self.k_weights.len() != self.q_weights.len()
            || self.v_weights.len() != self.q_weights.len()
            || self.o_weights.len() != self.q_weights.len()
            || self.up_weights.len() != up
            || self.gate_weights.len() != up
            || self.down_weights.len() != down
            || self.output_weights.len() != checked_product(config.vocab_size, config.d_model)?
            || self.output_bias_q8.len() != config.vocab_size
        {
            return Err(TrainError::InvalidModel("invalid production model shape"));
        }
        let actual_parameters = self
            .embeddings
            .len()
            .checked_add(self.attention_rms_weights.len())
            .and_then(|value| value.checked_add(self.mlp_rms_weights.len()))
            .and_then(|value| value.checked_add(self.final_rms_weights.len()))
            .and_then(|value| value.checked_add(self.q_weights.len()))
            .and_then(|value| value.checked_add(self.k_weights.len()))
            .and_then(|value| value.checked_add(self.v_weights.len()))
            .and_then(|value| value.checked_add(self.o_weights.len()))
            .and_then(|value| value.checked_add(self.up_weights.len()))
            .and_then(|value| value.checked_add(self.gate_weights.len()))
            .and_then(|value| value.checked_add(self.down_weights.len()))
            .and_then(|value| value.checked_add(self.output_weights.len()))
            .and_then(|value| value.checked_add(self.output_bias_q8.len()))
            .ok_or(TrainError::InvalidModel(
                "production parameter count overflow",
            ))?;
        if Some(actual_parameters) != config.parameter_count() {
            return Err(TrainError::InvalidModel(
                "production parameter count mismatch",
            ));
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count().unwrap_or(usize::MAX)
    }

    pub fn model_hash(&self) -> u64 {
        fnv1a(&self.bytes_without_checksum())
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        self.validate()?;
        let mut bytes = self.bytes_without_checksum();
        let checksum = fnv1a(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < 8 + 4 + 6 * 4 + 8 + 8 + 6 + 8 || &bytes[..8] != PRODUCTION_MODEL_V1_MAGIC {
            return Err(TrainError::InvalidModel("bad NSRLPM1 artifact magic"));
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(
            bytes[checksum_offset..]
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad NSRLPM1 checksum"))?,
        );
        if fnv1a(&bytes[..checksum_offset]) != expected_checksum {
            return Err(TrainError::InvalidModel("bad NSRLPM1 checksum"));
        }
        let mut offset = 8;
        if read_u32(bytes, &mut offset)? != PRODUCTION_MODEL_V1_VERSION {
            return Err(TrainError::InvalidModel("unsupported NSRLPM1 version"));
        }
        let config = ProductionModelConfig {
            vocab_size: read_u32(bytes, &mut offset)? as usize,
            d_model: read_u32(bytes, &mut offset)? as usize,
            heads: read_u32(bytes, &mut offset)? as usize,
            layers: read_u32(bytes, &mut offset)? as usize,
            hidden_dim: read_u32(bytes, &mut offset)? as usize,
            context_tokens: read_u32(bytes, &mut offset)? as usize,
        };
        config.validate()?;
        let tokenizer_hash = read_u64(bytes, &mut offset)?;
        let initialization_seed = read_u64(bytes, &mut offset)?;
        let shifts = take(bytes, &mut offset, 6)?;
        let scales = ProductionProjectionScales {
            qkv_shift: shifts[0],
            o_shift: shifts[1],
            up_shift: shifts[2],
            gate_shift: shifts[3],
            down_shift: shifts[4],
            output_shift: shifts[5],
        };
        let matrix = checked_product(config.d_model, config.d_model)?;
        let rms = checked_product(config.layers, config.d_model)?;
        let up = checked_product(
            config.layers,
            checked_product(config.d_model, config.hidden_dim)?,
        )?;
        let down = checked_product(
            config.layers,
            checked_product(config.hidden_dim, config.d_model)?,
        )?;
        let model = Self {
            config,
            tokenizer_hash,
            initialization_seed,
            scales,
            embeddings: read_i16(
                bytes,
                &mut offset,
                checked_product(config.vocab_size, config.d_model)?,
            )?,
            attention_rms_weights: read_i16(bytes, &mut offset, rms)?,
            mlp_rms_weights: read_i16(bytes, &mut offset, rms)?,
            final_rms_weights: read_i16(bytes, &mut offset, config.d_model)?,
            q_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            k_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            v_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            o_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            up_weights: read_i8(bytes, &mut offset, up)?,
            gate_weights: read_i8(bytes, &mut offset, up)?,
            down_weights: read_i8(bytes, &mut offset, down)?,
            output_weights: read_i16(
                bytes,
                &mut offset,
                checked_product(config.vocab_size, config.d_model)?,
            )?,
            output_bias_q8: read_i32(bytes, &mut offset, config.vocab_size)?,
        };
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel("wrong NSRLPM1 artifact length"));
        }
        model.validate()?;
        Ok(model)
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PRODUCTION_MODEL_V1_MAGIC);
        bytes.extend_from_slice(&PRODUCTION_MODEL_V1_VERSION.to_le_bytes());
        for value in [
            self.config.vocab_size,
            self.config.d_model,
            self.config.heads,
            self.config.layers,
            self.config.hidden_dim,
            self.config.context_tokens,
        ] {
            bytes.extend_from_slice(&(value as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&self.tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&self.initialization_seed.to_le_bytes());
        bytes.extend_from_slice(&scale_shifts(self.scales));
        extend_i16(&mut bytes, &self.embeddings);
        extend_i16(&mut bytes, &self.attention_rms_weights);
        extend_i16(&mut bytes, &self.mlp_rms_weights);
        extend_i16(&mut bytes, &self.final_rms_weights);
        extend_i8(&mut bytes, &self.q_weights);
        extend_i8(&mut bytes, &self.k_weights);
        extend_i8(&mut bytes, &self.v_weights);
        extend_i8(&mut bytes, &self.o_weights);
        extend_i8(&mut bytes, &self.up_weights);
        extend_i8(&mut bytes, &self.gate_weights);
        extend_i8(&mut bytes, &self.down_weights);
        extend_i16(&mut bytes, &self.output_weights);
        for value in &self.output_bias_q8 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

pub fn forward_production_model(
    model: &ProductionModelV1,
    context: &[u32],
) -> Result<ProductionForward, TrainError> {
    let (features_q15, residual_saturation_count) = production_features(model, context)?;
    let logits_q8 = output_logits(model, &features_q15)?;
    let mut probabilities_q15 = vec![0_i16; model.config.vocab_size];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("production_softmax"))?;
    Ok(ProductionForward {
        features_q15,
        logits_q8,
        probabilities_q15,
        residual_saturation_count,
    })
}

pub fn train_production_output_smoke(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionSmokeConfig,
) -> Result<ProductionSmokeTrace, TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_windows == 0
        || config.epochs == 0
        || config.feature_shift > 15
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, config.context_tokens, config.max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let initial_model_hash = model.model_hash();
    let mut cached = Vec::with_capacity(windows.len());
    let mut targets = Vec::with_capacity(windows.len());
    let mut residual_saturation_count = 0_usize;
    for (context, target) in &windows {
        let (features, saturation) = production_features(model, context)?;
        residual_saturation_count = residual_saturation_count.saturating_add(saturation);
        cached.push(features);
        targets.push(*target);
    }
    let initial_mistakes = count_mistakes(model, &cached, &targets)?;
    let mut updates = 0_usize;
    let mut weight_saturation_count = 0_usize;
    for _ in 0..config.epochs {
        for (features, &target) in cached.iter().zip(&targets) {
            let logits = output_logits(model, features)?;
            let predicted = argmax(&logits);
            let target_index = target as usize;
            let competitor = if predicted == target_index {
                argmax_except(&logits, target_index)
            } else {
                predicted
            };
            if predicted == target_index
                && logits[target_index] >= logits[competitor].saturating_add(config.margin_q8)
            {
                continue;
            }
            updates = updates.saturating_add(1);
            for (dim, &feature) in features.iter().enumerate() {
                let mut delta = i32::from(feature) >> config.feature_shift;
                if delta == 0 && feature != 0 {
                    delta = i32::from(feature.signum());
                }
                let target_offset = target_index * model.config.d_model + dim;
                let competitor_offset = competitor * model.config.d_model + dim;
                weight_saturation_count = weight_saturation_count
                    .saturating_add(update_i16(&mut model.output_weights[target_offset], delta));
                weight_saturation_count = weight_saturation_count.saturating_add(update_i16(
                    &mut model.output_weights[competitor_offset],
                    -delta,
                ));
            }
            model.output_bias_q8[target_index] =
                model.output_bias_q8[target_index].saturating_add(config.bias_step_q8);
            model.output_bias_q8[competitor] =
                model.output_bias_q8[competitor].saturating_sub(config.bias_step_q8);
        }
    }
    let final_mistakes = count_mistakes(model, &cached, &targets)?;
    Ok(ProductionSmokeTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        windows: windows.len(),
        epochs: config.epochs,
        initial_mistakes,
        final_mistakes,
        updates,
        weight_saturation_count,
        residual_saturation_count,
        initial_model_hash,
        final_model_hash: model.model_hash(),
    })
}

pub fn decode_bound_token_stream(
    bytes: &[u8],
    tokenizer_hash: u64,
    vocab_size: usize,
) -> Result<(Vec<u32>, u64), TrainError> {
    if bytes.len() < 24 || &bytes[..8] != b"NSRLTOK1" {
        return Err(TrainError::InvalidModel("bad NSRLTOK1 header"));
    }
    let artifact_hash = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad NSRLTOK1 tokenizer hash"))?,
    );
    if artifact_hash != tokenizer_hash {
        return Err(TrainError::InvalidModel("NSRLTOK1 tokenizer hash mismatch"));
    }
    let token_count = usize::try_from(u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad NSRLTOK1 token count"))?,
    ))
    .map_err(|_| TrainError::InvalidModel("NSRLTOK1 token count overflow"))?;
    if bytes.len() != 24_usize.saturating_add(token_count.saturating_mul(4)) {
        return Err(TrainError::InvalidModel("wrong NSRLTOK1 length"));
    }
    let mut tokens = Vec::with_capacity(token_count);
    for chunk in bytes[24..].chunks_exact(4) {
        let token = u32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| TrainError::InvalidModel("truncated NSRLTOK1 token"))?,
        );
        if token as usize >= vocab_size {
            return Err(TrainError::InvalidModel(
                "NSRLTOK1 token exceeds model vocabulary",
            ));
        }
        tokens.push(token);
    }
    Ok((tokens, fnv1a(&bytes[24..])))
}

fn production_features(
    model: &ProductionModelV1,
    context: &[u32],
) -> Result<(Vec<i16>, usize), TrainError> {
    model.validate()?;
    let config = model.config;
    if context.is_empty()
        || context.len() > config.context_tokens
        || context
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut hidden = Vec::with_capacity(context.len() * config.d_model);
    for &token in context {
        let start = token as usize * config.d_model;
        hidden.extend_from_slice(&model.embeddings[start..start + config.d_model]);
    }
    let seq_len = context.len();
    let total = checked_product(seq_len, config.d_model)?;
    let matrix = checked_product(config.d_model, config.d_model)?;
    let up_matrix = checked_product(config.d_model, config.hidden_dim)?;
    let down_matrix = checked_product(config.hidden_dim, config.d_model)?;
    let mut residual_saturation_count = 0_usize;
    let qkv_scales = scales(config.d_model, model.scales.qkv_shift);
    let o_scales = scales(config.d_model, model.scales.o_shift);
    let up_scales = scales(config.hidden_dim, model.scales.up_shift);
    let gate_scales = scales(config.hidden_dim, model.scales.gate_shift);
    let down_scales = scales(config.d_model, model.scales.down_shift);
    for layer in 0..config.layers {
        let rms = layer * config.d_model..(layer + 1) * config.d_model;
        let attention_input = rms_rows(
            &hidden,
            &model.attention_rms_weights[rms.clone()],
            config.d_model,
        )?;
        let range = layer * matrix..(layer + 1) * matrix;
        let params = SelfAttentionI16Params {
            q: linear_params(
                &model.q_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            k: linear_params(
                &model.k_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            v: linear_params(
                &model.v_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            o: linear_params(
                &model.o_weights[range],
                &o_scales,
                config.d_model,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            heads: config.heads,
            causal: true,
        };
        let head_dim = config.d_model / config.heads;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut attention_context = vec![0_i16; total];
        let mut state_kv = vec![0_i64; config.heads * head_dim * head_dim];
        let mut key_sums = vec![0_i64; config.heads * head_dim];
        let mut attention_output = vec![0_i16; total];
        linear_attention_i16_q15_checked(
            &attention_input,
            params,
            LinearAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut attention_context,
                state_kv: &mut state_kv,
                key_sums: &mut key_sums,
            },
            &mut attention_output,
        )
        .ok_or(TrainError::CoreRejected("production_linear_attention"))?;
        let mut attention_residual = vec![0_i16; total];
        residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
            &hidden,
            &attention_output,
            &mut attention_residual,
        ));
        let mlp_input = rms_rows(
            &attention_residual,
            &model.mlp_rms_weights[rms],
            config.d_model,
        )?;
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let mlp_params = GatedMlpI16Params {
            up: linear_params(
                &model.up_weights[up_range.clone()],
                &up_scales,
                config.d_model,
                config.hidden_dim,
            ),
            gate: linear_params(
                &model.gate_weights[up_range],
                &gate_scales,
                config.d_model,
                config.hidden_dim,
            ),
            down: linear_params(
                &model.down_weights[down_range],
                &down_scales,
                config.hidden_dim,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            hidden_dim: config.hidden_dim,
        };
        let mut up = vec![0_i16; seq_len * config.hidden_dim];
        let mut gate = vec![0_i16; seq_len * config.hidden_dim];
        let mut gated = vec![0_i16; seq_len * config.hidden_dim];
        let mut mlp_output = vec![0_i16; total];
        gated_mlp_i16_q15_checked(
            &mlp_input,
            mlp_params,
            GatedMlpWorkspace {
                up: &mut up,
                gate: &mut gate,
                gated: &mut gated,
            },
            &mut mlp_output,
        )
        .ok_or(TrainError::CoreRejected("production_gated_mlp"))?;
        residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
            &attention_residual,
            &mlp_output,
            &mut hidden,
        ));
    }
    let start = (seq_len - 1) * config.d_model;
    let mut features = vec![0_i16; config.d_model];
    rms_norm_i16_q15_checked(
        &hidden[start..start + config.d_model],
        &model.final_rms_weights,
        PRODUCTION_RMS_EPSILON,
        &mut features,
    )
    .ok_or(TrainError::CoreRejected("production_final_rms"))?;
    Ok((features, residual_saturation_count))
}

fn output_logits(model: &ProductionModelV1, features: &[i16]) -> Result<Vec<i32>, TrainError> {
    if features.len() != model.config.d_model {
        return Err(TrainError::InvalidConfig);
    }
    let mut logits = vec![0_i32; model.config.vocab_size];
    for (token, logit) in logits.iter_mut().enumerate() {
        let start = token * model.config.d_model;
        let mut accumulator = 0_i64;
        for (dim, &feature) in features.iter().enumerate() {
            accumulator = accumulator.saturating_add(
                i64::from(feature).saturating_mul(i64::from(model.output_weights[start + dim])),
            );
        }
        let shifted = accumulator >> model.scales.output_shift;
        *logit = (shifted.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
            .saturating_add(model.output_bias_q8[token]);
    }
    Ok(logits)
}

fn document_windows(
    tokens: &[u32],
    context_tokens: usize,
    max_windows: usize,
) -> Vec<(Vec<u32>, u32)> {
    let mut windows = Vec::new();
    let mut document = Vec::new();
    let mut in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document && document.len() > context_tokens {
                for start in 0..document.len() - context_tokens {
                    windows.push((
                        document[start..start + context_tokens].to_vec(),
                        document[start + context_tokens],
                    ));
                    if windows.len() >= max_windows {
                        return windows;
                    }
                }
            }
            document.clear();
            in_document = false;
        } else if in_document {
            document.push(token);
        }
    }
    windows
}

fn count_mistakes(
    model: &ProductionModelV1,
    features: &[Vec<i16>],
    targets: &[u32],
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for (features, &target) in features.iter().zip(targets) {
        mistakes = mistakes.saturating_add(usize::from(
            argmax(&output_logits(model, features)?) != target as usize,
        ));
    }
    Ok(mistakes)
}

fn rms_rows(input: &[i16], weights: &[i16], d_model: usize) -> Result<Vec<i16>, TrainError> {
    let mut output = vec![0_i16; input.len()];
    for (input, output) in input
        .chunks_exact(d_model)
        .zip(output.chunks_exact_mut(d_model))
    {
        rms_norm_i16_q15_checked(input, weights, PRODUCTION_RMS_EPSILON, output)
            .ok_or(TrainError::CoreRejected("production_rms"))?;
    }
    Ok(output)
}

fn add_residual(left: &[i16], right: &[i16], output: &mut [i16]) -> usize {
    let mut saturation = 0_usize;
    for index in 0..output.len() {
        let value = i64::from(left[index]) + i64::from(right[index]);
        saturation = saturation.saturating_add(usize::from(
            value < i64::from(i16::MIN) || value > i64::from(i16::MAX),
        ));
        output[index] = saturate_i16(value);
    }
    saturation
}

fn linear_params<'a>(
    weights: &'a [i8],
    scales: &'a [FixedScale],
    input_dim: usize,
    output_dim: usize,
) -> LinearI16I8Params<'a> {
    LinearI16I8Params {
        weights,
        bias: None,
        scales,
        input_dim,
        output_dim,
    }
}

fn scales(count: usize, shift: u8) -> Vec<FixedScale> {
    vec![
        FixedScale {
            multiplier: 1,
            right_shift: shift,
        };
        count
    ]
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

fn stacked_identity(layers: usize, d_model: usize, strength: i8) -> Vec<i8> {
    let mut values = vec![0_i8; layers * d_model * d_model];
    for layer in 0..layers {
        let start = layer * d_model * d_model;
        for dim in 0..d_model {
            values[start + dim * d_model + dim] = strength;
        }
    }
    values
}

fn initial_i8_tensor(count: usize, seed: u64, amplitude: i8) -> Vec<i8> {
    let span = i16::from(amplitude) * 2 + 1;
    (0..count)
        .map(|index| {
            let value = splitmix64(seed ^ index as u64);
            ((value % span as u64) as i16 - i16::from(amplitude)) as i8
        })
        .collect()
}

fn initial_i16_tensor(count: usize, seed: u64, amplitude: i16) -> Vec<i16> {
    let span = i32::from(amplitude) * 2 + 1;
    (0..count)
        .map(|index| {
            let value = splitmix64(seed ^ index as u64) % span as u64;
            (value as i32 - i32::from(amplitude)) as i16
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn argmax(values: &[i32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn argmax_except(values: &[i32], excluded: usize) -> usize {
    values
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != excluded)
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn update_i16(value: &mut i16, delta: i32) -> usize {
    let updated = i64::from(*value) + i64::from(delta);
    let saturated = updated < i64::from(i16::MIN) || updated > i64::from(i16::MAX);
    *value = saturate_i16(updated);
    usize::from(saturated)
}

fn checked_product(left: usize, right: usize) -> Result<usize, TrainError> {
    left.checked_mul(right).ok_or(TrainError::InvalidConfig)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn extend_i8(bytes: &mut Vec<u8>, values: &[i8]) {
    bytes.extend(values.iter().map(|&value| value as u8));
}

fn extend_i16(bytes: &mut Vec<u8>, values: &[i16]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, count: usize) -> Result<&'a [u8], TrainError> {
    let end = offset.checked_add(count).ok_or(TrainError::InvalidModel(
        "production artifact offset overflow",
    ))?;
    let value = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("truncated production artifact"))?;
    *offset = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, TrainError> {
    Ok(u32::from_le_bytes(
        take(bytes, offset, 4)?
            .try_into()
            .map_err(|_| TrainError::InvalidModel("truncated production u32"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, TrainError> {
    Ok(u64::from_le_bytes(
        take(bytes, offset, 8)?
            .try_into()
            .map_err(|_| TrainError::InvalidModel("truncated production u64"))?,
    ))
}

fn read_i8(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, TrainError> {
    Ok(take(bytes, offset, count)?
        .iter()
        .map(|&value| value as i8)
        .collect())
}

fn read_i16(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, TrainError> {
    let raw = take(
        bytes,
        offset,
        count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("production i16 length overflow"))?,
    )?;
    Ok(raw
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_i32(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i32>, TrainError> {
    let raw = take(
        bytes,
        offset,
        count
            .checked_mul(4)
            .ok_or(TrainError::InvalidModel("production i32 length overflow"))?,
    )?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ProductionModelConfig {
        ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 2,
            hidden_dim: 48,
            context_tokens: 16,
        }
    }

    #[test]
    fn frozen_profiles_match_scaling_plan_parameter_counts() {
        assert_eq!(
            ProductionModelConfig::profile("p10m")
                .unwrap()
                .parameter_count(),
            Some(9_317_632)
        );
        assert_eq!(
            ProductionModelConfig::profile("p20m")
                .unwrap()
                .parameter_count(),
            Some(21_641_600)
        );
        assert_eq!(
            ProductionModelConfig::profile("p30m")
                .unwrap()
                .parameter_count(),
            Some(28_229_056)
        );
    }

    #[test]
    fn production_artifact_round_trips_and_binds_tokenizer() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 7).expect("model");
        let bytes = model.try_to_bytes().expect("serialize");
        assert_eq!(
            ProductionModelV1::from_bytes(&bytes).expect("decode"),
            model
        );
        let mut corrupt = bytes;
        corrupt[32] ^= 1;
        assert!(ProductionModelV1::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn production_forward_and_smoke_training_accept_u32_tokens() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let forward = forward_production_model(&model, &[300, 301, 302, 303]).expect("forward");
        assert_eq!(forward.logits_q8.len(), 320);
        assert_eq!(forward.probabilities_q15.len(), 320);
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace = train_production_output_smoke(
            &mut model,
            &tokens,
            0x5678,
            ProductionSmokeConfig {
                context_tokens: 4,
                max_windows: 2,
                epochs: 4,
                ..ProductionSmokeConfig::default()
            },
        )
        .expect("smoke train");
        assert_eq!(trace.windows, 2);
        assert_ne!(trace.initial_model_hash, trace.final_model_hash);
        assert!(trace.final_mistakes <= trace.initial_mistakes);
    }

    #[test]
    fn token_stream_loader_rejects_wrong_binding_and_vocab() {
        let tokenizer_hash = 0x1234_u64;
        let tokens = [BOS_TOKEN_ID, 300, EOS_TOKEN_ID];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NSRLTOK1");
        bytes.extend_from_slice(&tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        assert_eq!(
            decode_bound_token_stream(&bytes, tokenizer_hash, 320)
                .unwrap()
                .0,
            tokens
        );
        assert!(decode_bound_token_stream(&bytes, tokenizer_hash + 1, 320).is_err());
        assert!(decode_bound_token_stream(&bytes, tokenizer_hash, 300).is_err());
    }

    #[test]
    fn production_full_backward_moves_every_group_and_resumes_bound_state() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 19).expect("model");
        let tokens = [
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            EOS_TOKEN_ID,
        ];
        let config = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 4,
            epochs: 2,
            ..ProductionFullTrainConfig::default()
        };
        let (first, state) = train_production_full_smoke(&mut model, &tokens, 0x5678, config, None)
            .expect("full train");
        assert!(
            first.movement_l1.iter().all(|&movement| movement > 0),
            "movement: {:?}",
            first.movement_l1
        );
        assert_ne!(first.initial_model_hash, first.final_model_hash);
        let bytes = state.try_to_bytes().expect("optimizer bytes");
        let decoded = ProductionOptimizerStateV1::from_bytes(&bytes).expect("optimizer decode");
        assert_eq!(decoded, state);
        let prior_step = state.step;
        let (_, resumed) =
            train_production_full_smoke(&mut model, &tokens, 0x5678, config, Some(decoded))
                .expect("resume");
        assert_eq!(resumed.step, prior_step + first.optimizer_steps as u64);
        assert!(
            train_production_full_smoke(&mut model, &tokens, 0x5679, config, Some(resumed))
                .is_err()
        );
    }
}
