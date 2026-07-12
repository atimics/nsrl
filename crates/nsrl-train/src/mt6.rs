//! NSRLMT6: a calibrated integer transformer with an explicit local-order path.
//!
//! MT6 is intentionally separate from `MiniTransformerMlpModel`: MT5 decoding,
//! hashes, and frozen proof-v1 semantics remain unchanged.

use nsrl_core::{
    FixedScale, GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionWorkspace, LinearI16I8Params,
    SelfAttentionI16Params, base2_softmax_i32_q15, gated_mlp_i16_q15_checked,
    linear_attention_i16_q15_checked, linear_i16_i8_i16_per_channel_checked,
    rms_norm_i16_q15_checked, saturate_i16,
};

use crate::{BYTE_VOCAB, MINI_TRANSFORMER_V6_MODEL_MAGIC, TrainError};

pub const MT6_D_MODEL: usize = 128;
pub const MT6_HEADS: usize = 8;
pub const MT6_HIDDEN_DIM: usize = 256;
pub const MT6_LAYERS: usize = 2;
pub const MT6_LOCAL_TAPS: usize = 4;
pub const MT6_FEATURE_DIM: usize = MT6_D_MODEL * (MT6_LOCAL_TAPS + 1);
pub const MT6_ARTIFACT_VERSION: u32 = 1;
pub const MT6_RMS_EPSILON: u64 = 1;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mt6ProjectionScales {
    pub qkv_shift: u8,
    pub o_shift: u8,
    pub up_shift: u8,
    pub gate_shift: u8,
    pub down_shift: u8,
    pub local_shift: u8,
    pub output_shift: u8,
}

impl Default for Mt6ProjectionScales {
    fn default() -> Self {
        Self {
            qkv_shift: 4,
            o_shift: 4,
            up_shift: 5,
            gate_shift: 5,
            down_shift: 6,
            local_shift: 4,
            output_shift: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerV6Model {
    pub context_seq_len: usize,
    pub scales: Mt6ProjectionScales,
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
    pub local_weights: Vec<i8>,
    pub output_weights: Vec<i16>,
    pub output_bias_q8: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt6Forward {
    pub features_q15: [i16; MT6_FEATURE_DIM],
    pub logits_q8: [i32; BYTE_VOCAB],
    pub probabilities_q15: [i16; BYTE_VOCAB],
    pub residual_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mt6OverfitConfig {
    pub epochs: usize,
    pub feature_shift: u8,
    pub bias_step_q8: i32,
    pub margin_q8: i32,
}

impl Default for Mt6OverfitConfig {
    fn default() -> Self {
        Self {
            epochs: 64,
            feature_shift: 13,
            bias_step_q8: 4,
            margin_q8: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mt6OverfitTrace {
    pub windows: usize,
    pub epochs: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub final_accuracy_per_mille: usize,
    pub updates: usize,
    pub weight_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
}

impl Mt6OverfitTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"nsrl.mt6.local_overfit.v1\",",
                "\"architecture\":{{\"magic\":\"NSRLMT6\",\"d_model\":{},\"heads\":{},\"hidden_dim\":{},\"layers\":{},\"local_taps\":{},\"feature_dim\":{},\"output_head_bits\":16}},",
                "\"training\":{{\"windows\":{},\"epochs\":{},\"updates\":{}}},",
                "\"evaluation\":{{\"initial_mistakes\":{},\"final_mistakes\":{},\"final_accuracy_per_mille\":{}}},",
                "\"health\":{{\"weight_saturation_count\":{},\"residual_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\"}}}}\n"
            ),
            MT6_D_MODEL,
            MT6_HEADS,
            MT6_HIDDEN_DIM,
            MT6_LAYERS,
            MT6_LOCAL_TAPS,
            MT6_FEATURE_DIM,
            self.windows,
            self.epochs,
            self.updates,
            self.initial_mistakes,
            self.final_mistakes,
            self.final_accuracy_per_mille,
            self.weight_saturation_count,
            self.residual_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
        )
    }
}

impl MiniTransformerV6Model {
    pub fn new_initial(context_seq_len: usize) -> Result<Self, TrainError> {
        if context_seq_len < MT6_LOCAL_TAPS {
            return Err(TrainError::InvalidConfig);
        }
        let matrix = MT6_D_MODEL * MT6_D_MODEL;
        let mlp_matrix = MT6_D_MODEL * MT6_HIDDEN_DIM;
        let model = Self {
            context_seq_len,
            scales: Mt6ProjectionScales::default(),
            embeddings: initial_embeddings(),
            attention_rms_weights: vec![i16::MAX; MT6_LAYERS * MT6_D_MODEL],
            mlp_rms_weights: vec![i16::MAX; MT6_LAYERS * MT6_D_MODEL],
            final_rms_weights: vec![i16::MAX; MT6_FEATURE_DIM],
            q_weights: stacked_identity(MT6_LAYERS, 16),
            k_weights: stacked_identity(MT6_LAYERS, 16),
            v_weights: stacked_identity(MT6_LAYERS, 16),
            o_weights: stacked_identity(MT6_LAYERS, 8),
            up_weights: initial_i8_tensor(MT6_LAYERS * mlp_matrix, 0x510e_527f_ade6_82d1, 2),
            gate_weights: initial_i8_tensor(MT6_LAYERS * mlp_matrix, 0x9b05_688c_2b3e_6c1f, 2),
            down_weights: initial_i8_tensor(
                MT6_LAYERS * MT6_HIDDEN_DIM * MT6_D_MODEL,
                0x1f83_d9ab_fb41_bd6b,
                1,
            ),
            local_weights: initial_local_weights(),
            output_weights: vec![0_i16; BYTE_VOCAB * MT6_FEATURE_DIM],
            output_bias_q8: vec![0_i32; BYTE_VOCAB],
        };
        debug_assert_eq!(model.q_weights.len(), MT6_LAYERS * matrix);
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), TrainError> {
        if u32::try_from(self.context_seq_len).is_err() {
            return Err(TrainError::InvalidModel(
                "NSRLMT6 context length exceeds u32",
            ));
        }
        let matrix = MT6_D_MODEL * MT6_D_MODEL;
        let rms = MT6_LAYERS * MT6_D_MODEL;
        let up = MT6_LAYERS * MT6_D_MODEL * MT6_HIDDEN_DIM;
        let down = MT6_LAYERS * MT6_HIDDEN_DIM * MT6_D_MODEL;
        if self.context_seq_len < MT6_LOCAL_TAPS
            || self.embeddings.len() != BYTE_VOCAB * MT6_D_MODEL
            || self.attention_rms_weights.len() != rms
            || self.mlp_rms_weights.len() != rms
            || self.final_rms_weights.len() != MT6_FEATURE_DIM
            || self.q_weights.len() != MT6_LAYERS * matrix
            || self.k_weights.len() != self.q_weights.len()
            || self.v_weights.len() != self.q_weights.len()
            || self.o_weights.len() != self.q_weights.len()
            || self.up_weights.len() != up
            || self.gate_weights.len() != up
            || self.down_weights.len() != down
            || self.local_weights.len() != MT6_LOCAL_TAPS * matrix
            || self.output_weights.len() != BYTE_VOCAB * MT6_FEATURE_DIM
            || self.output_bias_q8.len() != BYTE_VOCAB
            || scale_shifts(self.scales).iter().any(|&shift| shift > 30)
        {
            return Err(TrainError::InvalidModel("invalid NSRLMT6 shape"));
        }
        Ok(())
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
        if bytes.len() < 8 + 4 + 4 + 7 + 8 || &bytes[..8] != MINI_TRANSFORMER_V6_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad NSRLMT6 artifact magic"));
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(
            bytes[checksum_offset..]
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad NSRLMT6 checksum"))?,
        );
        if fnv1a(&bytes[..checksum_offset]) != expected_checksum {
            return Err(TrainError::InvalidModel("bad NSRLMT6 checksum"));
        }
        let mut offset = 8;
        let version = read_u32(bytes, &mut offset)?;
        if version != MT6_ARTIFACT_VERSION {
            return Err(TrainError::InvalidModel("unsupported NSRLMT6 version"));
        }
        let context_seq_len = read_u32(bytes, &mut offset)? as usize;
        let shifts = take(bytes, &mut offset, 7)?;
        let scales = Mt6ProjectionScales {
            qkv_shift: shifts[0],
            o_shift: shifts[1],
            up_shift: shifts[2],
            gate_shift: shifts[3],
            down_shift: shifts[4],
            local_shift: shifts[5],
            output_shift: shifts[6],
        };
        let model = Self {
            context_seq_len,
            scales,
            embeddings: read_i16(bytes, &mut offset, BYTE_VOCAB * MT6_D_MODEL)?,
            attention_rms_weights: read_i16(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL)?,
            mlp_rms_weights: read_i16(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL)?,
            final_rms_weights: read_i16(bytes, &mut offset, MT6_FEATURE_DIM)?,
            q_weights: read_i8(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL * MT6_D_MODEL)?,
            k_weights: read_i8(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL * MT6_D_MODEL)?,
            v_weights: read_i8(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL * MT6_D_MODEL)?,
            o_weights: read_i8(bytes, &mut offset, MT6_LAYERS * MT6_D_MODEL * MT6_D_MODEL)?,
            up_weights: read_i8(
                bytes,
                &mut offset,
                MT6_LAYERS * MT6_D_MODEL * MT6_HIDDEN_DIM,
            )?,
            gate_weights: read_i8(
                bytes,
                &mut offset,
                MT6_LAYERS * MT6_D_MODEL * MT6_HIDDEN_DIM,
            )?,
            down_weights: read_i8(
                bytes,
                &mut offset,
                MT6_LAYERS * MT6_HIDDEN_DIM * MT6_D_MODEL,
            )?,
            local_weights: read_i8(
                bytes,
                &mut offset,
                MT6_LOCAL_TAPS * MT6_D_MODEL * MT6_D_MODEL,
            )?,
            output_weights: read_i16(bytes, &mut offset, BYTE_VOCAB * MT6_FEATURE_DIM)?,
            output_bias_q8: read_i32(bytes, &mut offset, BYTE_VOCAB)?,
        };
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel("wrong NSRLMT6 artifact length"));
        }
        model.validate()?;
        Ok(model)
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MINI_TRANSFORMER_V6_MODEL_MAGIC);
        bytes.extend_from_slice(&MT6_ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.context_seq_len as u32).to_le_bytes());
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
        extend_i8(&mut bytes, &self.local_weights);
        extend_i16(&mut bytes, &self.output_weights);
        for value in &self.output_bias_q8 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

pub fn forward_mt6(
    model: &MiniTransformerV6Model,
    context: &[u8],
) -> Result<Mt6Forward, TrainError> {
    let features = mt6_features(model, context)?;
    output_from_features(model, features.0, features.1)
}

pub fn train_mt6_output_overfit(
    model: &mut MiniTransformerV6Model,
    contexts: &[Vec<u8>],
    targets: &[u8],
    config: Mt6OverfitConfig,
) -> Result<Mt6OverfitTrace, TrainError> {
    model.validate()?;
    if contexts.is_empty()
        || contexts.len() != targets.len()
        || config.epochs == 0
        || config.feature_shift > 15
    {
        return Err(TrainError::InvalidConfig);
    }
    let initial_model_hash = model.model_hash();
    let mut cached = Vec::with_capacity(contexts.len());
    let mut residual_saturation_count = 0_usize;
    for context in contexts {
        let (features, saturation) = mt6_features(model, context)?;
        residual_saturation_count = residual_saturation_count.saturating_add(saturation);
        cached.push(features);
    }
    let initial_mistakes = count_feature_mistakes(model, &cached, targets)?;
    let mut updates = 0_usize;
    let mut weight_saturation_count = 0_usize;
    for _ in 0..config.epochs {
        for (features, &target) in cached.iter().zip(targets) {
            let output = output_from_features(model, *features, 0)?;
            let predicted = argmax(&output.logits_q8);
            let target_index = usize::from(target);
            let competitor = if predicted == target {
                argmax_except(&output.logits_q8, target)
            } else {
                predicted
            };
            let competitor_index = usize::from(competitor);
            let needs_update = predicted != target
                || output.logits_q8[target_index]
                    < output.logits_q8[competitor_index].saturating_add(config.margin_q8);
            if !needs_update {
                continue;
            }
            updates = updates.saturating_add(1);
            for (dim, &feature) in features.iter().enumerate() {
                let mut delta = i32::from(feature) >> config.feature_shift;
                if delta == 0 && feature != 0 {
                    delta = i32::from(feature.signum());
                }
                let target_offset = target_index * MT6_FEATURE_DIM + dim;
                let competitor_offset = competitor_index * MT6_FEATURE_DIM + dim;
                weight_saturation_count = weight_saturation_count
                    .saturating_add(update_i16(&mut model.output_weights[target_offset], delta));
                weight_saturation_count = weight_saturation_count.saturating_add(update_i16(
                    &mut model.output_weights[competitor_offset],
                    -delta,
                ));
            }
            model.output_bias_q8[target_index] =
                model.output_bias_q8[target_index].saturating_add(config.bias_step_q8);
            model.output_bias_q8[competitor_index] =
                model.output_bias_q8[competitor_index].saturating_sub(config.bias_step_q8);
        }
    }
    let final_mistakes = count_feature_mistakes(model, &cached, targets)?;
    Ok(Mt6OverfitTrace {
        windows: contexts.len(),
        epochs: config.epochs,
        initial_mistakes,
        final_mistakes,
        final_accuracy_per_mille: contexts.len().saturating_sub(final_mistakes) * 1000
            / contexts.len(),
        updates,
        weight_saturation_count,
        residual_saturation_count,
        initial_model_hash,
        final_model_hash: model.model_hash(),
    })
}

fn mt6_features(
    model: &MiniTransformerV6Model,
    context: &[u8],
) -> Result<([i16; MT6_FEATURE_DIM], usize), TrainError> {
    if context.len() != model.context_seq_len || context.len() < MT6_LOCAL_TAPS {
        return Err(TrainError::InvalidConfig);
    }
    let mut hidden = Vec::with_capacity(context.len() * MT6_D_MODEL);
    for &token in context {
        let start = usize::from(token) * MT6_D_MODEL;
        hidden.extend_from_slice(&model.embeddings[start..start + MT6_D_MODEL]);
    }
    let seq_len = context.len();
    let total = seq_len * MT6_D_MODEL;
    let matrix = MT6_D_MODEL * MT6_D_MODEL;
    let up_matrix = MT6_D_MODEL * MT6_HIDDEN_DIM;
    let down_matrix = MT6_HIDDEN_DIM * MT6_D_MODEL;
    let mut residual_saturation_count = 0_usize;
    for layer in 0..MT6_LAYERS {
        let rms = layer * MT6_D_MODEL..(layer + 1) * MT6_D_MODEL;
        let attention_input = rms_rows(&hidden, &model.attention_rms_weights[rms.clone()])?;
        let qkv_scales = scales(MT6_D_MODEL, model.scales.qkv_shift);
        let o_scales = scales(MT6_D_MODEL, model.scales.o_shift);
        let range = layer * matrix..(layer + 1) * matrix;
        let params = SelfAttentionI16Params {
            q: linear_params(
                &model.q_weights[range.clone()],
                &qkv_scales,
                MT6_D_MODEL,
                MT6_D_MODEL,
            ),
            k: linear_params(
                &model.k_weights[range.clone()],
                &qkv_scales,
                MT6_D_MODEL,
                MT6_D_MODEL,
            ),
            v: linear_params(
                &model.v_weights[range.clone()],
                &qkv_scales,
                MT6_D_MODEL,
                MT6_D_MODEL,
            ),
            o: linear_params(&model.o_weights[range], &o_scales, MT6_D_MODEL, MT6_D_MODEL),
            seq_len,
            d_model: MT6_D_MODEL,
            heads: MT6_HEADS,
            causal: true,
        };
        let head_dim = MT6_D_MODEL / MT6_HEADS;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut attention_context = vec![0_i16; total];
        let mut state_kv = vec![0_i64; MT6_HEADS * head_dim * head_dim];
        let mut key_sums = vec![0_i64; MT6_HEADS * head_dim];
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
        .ok_or(TrainError::CoreRejected("mt6_linear_attention"))?;
        let mut attention_residual = vec![0_i16; total];
        residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
            &hidden,
            &attention_output,
            &mut attention_residual,
        ));
        let mlp_input = rms_rows(&attention_residual, &model.mlp_rms_weights[rms])?;
        let up_scales = scales(MT6_HIDDEN_DIM, model.scales.up_shift);
        let gate_scales = scales(MT6_HIDDEN_DIM, model.scales.gate_shift);
        let down_scales = scales(MT6_D_MODEL, model.scales.down_shift);
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let mlp_params = GatedMlpI16Params {
            up: linear_params(
                &model.up_weights[up_range.clone()],
                &up_scales,
                MT6_D_MODEL,
                MT6_HIDDEN_DIM,
            ),
            gate: linear_params(
                &model.gate_weights[up_range],
                &gate_scales,
                MT6_D_MODEL,
                MT6_HIDDEN_DIM,
            ),
            down: linear_params(
                &model.down_weights[down_range],
                &down_scales,
                MT6_HIDDEN_DIM,
                MT6_D_MODEL,
            ),
            seq_len,
            d_model: MT6_D_MODEL,
            hidden_dim: MT6_HIDDEN_DIM,
        };
        let mut up = vec![0_i16; seq_len * MT6_HIDDEN_DIM];
        let mut gate = vec![0_i16; seq_len * MT6_HIDDEN_DIM];
        let mut gated = vec![0_i16; seq_len * MT6_HIDDEN_DIM];
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
        .ok_or(TrainError::CoreRejected("mt6_gated_mlp"))?;
        residual_saturation_count = residual_saturation_count.saturating_add(add_residual(
            &attention_residual,
            &mlp_output,
            &mut hidden,
        ));
    }

    let mut combined = [0_i16; MT6_FEATURE_DIM];
    let local_scales = scales(MT6_D_MODEL, model.scales.local_shift);
    for tap in 0..MT6_LOCAL_TAPS {
        let token = context[context.len() - 1 - tap];
        let embedding_start = usize::from(token) * MT6_D_MODEL;
        let weight_start = tap * matrix;
        let mut projected = [0_i16; MT6_D_MODEL];
        linear_i16_i8_i16_per_channel_checked(
            &model.embeddings[embedding_start..embedding_start + MT6_D_MODEL],
            linear_params(
                &model.local_weights[weight_start..weight_start + matrix],
                &local_scales,
                MT6_D_MODEL,
                MT6_D_MODEL,
            ),
            &mut projected,
        )
        .ok_or(TrainError::CoreRejected("mt6_local_projection"))?;
        let local_start = MT6_D_MODEL + tap * MT6_D_MODEL;
        combined[local_start..local_start + MT6_D_MODEL].copy_from_slice(&projected);
    }
    let final_start = (seq_len - 1) * MT6_D_MODEL;
    for dim in 0..MT6_D_MODEL {
        combined[dim] = hidden[final_start + dim] >> 2;
    }
    let mut features = [0_i16; MT6_FEATURE_DIM];
    rms_norm_i16_q15_checked(
        &combined,
        &model.final_rms_weights,
        MT6_RMS_EPSILON,
        &mut features,
    )
    .ok_or(TrainError::CoreRejected("mt6_final_rms"))?;
    Ok((features, residual_saturation_count))
}

fn output_from_features(
    model: &MiniTransformerV6Model,
    features_q15: [i16; MT6_FEATURE_DIM],
    residual_saturation_count: usize,
) -> Result<Mt6Forward, TrainError> {
    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (token, logit) in logits_q8.iter_mut().enumerate() {
        let start = token * MT6_FEATURE_DIM;
        let mut accumulator = 0_i64;
        for (dim, &feature) in features_q15.iter().enumerate() {
            accumulator = accumulator.saturating_add(
                i64::from(feature).saturating_mul(i64::from(model.output_weights[start + dim])),
            );
        }
        let shifted = accumulator >> model.scales.output_shift;
        *logit = (shifted.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
            .saturating_add(model.output_bias_q8[token]);
    }
    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("mt6_softmax"))?;
    Ok(Mt6Forward {
        features_q15,
        logits_q8,
        probabilities_q15,
        residual_saturation_count,
    })
}

fn count_feature_mistakes(
    model: &MiniTransformerV6Model,
    features: &[[i16; MT6_FEATURE_DIM]],
    targets: &[u8],
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for (features, &target) in features.iter().zip(targets) {
        let output = output_from_features(model, *features, 0)?;
        mistakes = mistakes.saturating_add(usize::from(argmax(&output.logits_q8) != target));
    }
    Ok(mistakes)
}

fn rms_rows(input: &[i16], weights: &[i16]) -> Result<Vec<i16>, TrainError> {
    let mut output = vec![0_i16; input.len()];
    for (input, output) in input
        .chunks_exact(MT6_D_MODEL)
        .zip(output.chunks_exact_mut(MT6_D_MODEL))
    {
        rms_norm_i16_q15_checked(input, weights, MT6_RMS_EPSILON, output)
            .ok_or(TrainError::CoreRejected("mt6_rms"))?;
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

fn scale_shifts(scales: Mt6ProjectionScales) -> [u8; 7] {
    [
        scales.qkv_shift,
        scales.o_shift,
        scales.up_shift,
        scales.gate_shift,
        scales.down_shift,
        scales.local_shift,
        scales.output_shift,
    ]
}

fn initial_embeddings() -> Vec<i16> {
    let mut values = Vec::with_capacity(BYTE_VOCAB * MT6_D_MODEL);
    for index in 0..BYTE_VOCAB * MT6_D_MODEL {
        let bucket = mixed_bucket(index, 0x6a09_e667_f3bc_c909, 32);
        values.push((bucket * 128) as i16);
    }
    values
}

fn initial_i8_tensor(len: usize, seed: u64, radius: u64) -> Vec<i8> {
    (0..len)
        .map(|index| mixed_bucket(index, seed, radius) as i8)
        .collect()
}

fn stacked_identity(layers: usize, diagonal: i8) -> Vec<i8> {
    let mut values = vec![0_i8; layers * MT6_D_MODEL * MT6_D_MODEL];
    for layer in 0..layers {
        let start = layer * MT6_D_MODEL * MT6_D_MODEL;
        for dim in 0..MT6_D_MODEL {
            values[start + dim * MT6_D_MODEL + dim] = diagonal;
        }
    }
    values
}

fn initial_local_weights() -> Vec<i8> {
    let matrix = MT6_D_MODEL * MT6_D_MODEL;
    let mut weights = vec![0_i8; MT6_LOCAL_TAPS * matrix];
    for tap in 0..MT6_LOCAL_TAPS {
        for output in 0..MT6_D_MODEL {
            for lane in 0..4 {
                let input = (output * 29 + lane * 31 + tap * 17) % MT6_D_MODEL;
                let sign = if (output + lane + tap).is_multiple_of(2) {
                    16
                } else {
                    -16
                };
                weights[tap * matrix + output * MT6_D_MODEL + input] = sign;
            }
        }
    }
    weights
}

fn mixed_bucket(index: usize, seed: u64, radius: u64) -> i32 {
    let mut value = (index as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value % (radius * 2 + 1)) as i32 - radius as i32
}

fn update_i16(weight: &mut i16, delta: i32) -> usize {
    let wide = i32::from(*weight).saturating_add(delta);
    let clipped = wide.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    *weight = clipped as i16;
    usize::from(wide != clipped)
}

fn argmax(logits: &[i32; BYTE_VOCAB]) -> u8 {
    logits
        .iter()
        .enumerate()
        .max_by_key(|&(token, logit)| (*logit, core::cmp::Reverse(token)))
        .map(|(token, _)| token as u8)
        .unwrap_or(0)
}

fn argmax_except(logits: &[i32; BYTE_VOCAB], excluded: u8) -> u8 {
    logits
        .iter()
        .enumerate()
        .filter(|&(token, _)| token != usize::from(excluded))
        .max_by_key(|&(token, logit)| (*logit, core::cmp::Reverse(token)))
        .map(|(token, _)| token as u8)
        .unwrap_or(excluded)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn extend_i16(bytes: &mut Vec<u8>, values: &[i16]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn extend_i8(bytes: &mut Vec<u8>, values: &[i8]) {
    bytes.extend(values.iter().map(|&value| value as u8));
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8], TrainError> {
    let end = offset
        .checked_add(len)
        .ok_or(TrainError::InvalidModel("NSRLMT6 offset overflow"))?;
    let value = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("truncated NSRLMT6 artifact"))?;
    *offset = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, TrainError> {
    Ok(u32::from_le_bytes(
        take(bytes, offset, 4)?
            .try_into()
            .map_err(|_| TrainError::InvalidModel("truncated NSRLMT6 u32"))?,
    ))
}

fn read_i8(bytes: &[u8], offset: &mut usize, len: usize) -> Result<Vec<i8>, TrainError> {
    Ok(take(bytes, offset, len)?
        .iter()
        .map(|&value| value as i8)
        .collect())
}

fn read_i16(bytes: &[u8], offset: &mut usize, len: usize) -> Result<Vec<i16>, TrainError> {
    let raw = take(bytes, offset, len * 2)?;
    Ok(raw
        .chunks_exact(2)
        .map(|value| i16::from_le_bytes([value[0], value[1]]))
        .collect())
}

fn read_i32(bytes: &[u8], offset: &mut usize, len: usize) -> Result<Vec<i32>, TrainError> {
    let raw = take(bytes, offset, len * 4)?;
    Ok(raw
        .chunks_exact(4)
        .map(|value| i32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MINI_TRANSFORMER_MODEL_MAGIC, MiniTransformerMlpModel};
    use std::collections::BTreeSet;

    #[test]
    fn mt6_round_trips_and_does_not_decode_as_mt5() {
        let model = MiniTransformerV6Model::new_initial(8).expect("model");
        let bytes = model.try_to_bytes().expect("bytes");
        assert_eq!(&bytes[..8], MINI_TRANSFORMER_V6_MODEL_MAGIC);
        assert_ne!(&bytes[..8], MINI_TRANSFORMER_MODEL_MAGIC);
        assert_eq!(
            MiniTransformerV6Model::from_bytes(&bytes).expect("decode"),
            model
        );
        assert!(MiniTransformerMlpModel::from_bytes(&bytes).is_err());
        assert!(
            MiniTransformerV6Model::from_bytes(
                &MiniTransformerMlpModel::new_initial_with_seq_len(8).to_bytes()
            )
            .is_err()
        );
        let mut corrupt = bytes;
        let index = corrupt.len() / 2;
        corrupt[index] ^= 0x40;
        assert!(MiniTransformerV6Model::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn mt6_initial_byte_embeddings_are_unique() {
        let model = MiniTransformerV6Model::new_initial(8).expect("model");
        let rows = model
            .embeddings
            .chunks_exact(MT6_D_MODEL)
            .collect::<BTreeSet<_>>();
        assert_eq!(rows.len(), BYTE_VOCAB);
    }

    #[test]
    fn mt6_forward_is_deterministic_and_calibrated() {
        let model = MiniTransformerV6Model::new_initial(8).expect("model");
        let left = forward_mt6(&model, b"abcdefgh").expect("left");
        let right = forward_mt6(&model, b"abcdefgh").expect("right");
        assert_eq!(left, right);
        assert!(left.residual_saturation_count < 8 * MT6_D_MODEL);
    }

    #[test]
    fn mt6_output_gate_can_memorize_small_local_fixture() {
        let mut model = MiniTransformerV6Model::new_initial(8).expect("model");
        let contexts = (0_u8..32)
            .map(|index| {
                let mut context = b"prefix__".to_vec();
                context[6] = index;
                context[7] = index.wrapping_mul(17);
                context
            })
            .collect::<Vec<_>>();
        let targets = (0_u8..32)
            .map(|index| index.wrapping_mul(7))
            .collect::<Vec<u8>>();
        let trace = train_mt6_output_overfit(
            &mut model,
            &contexts,
            &targets,
            Mt6OverfitConfig {
                epochs: 96,
                ..Mt6OverfitConfig::default()
            },
        )
        .expect("train");
        assert!(trace.final_accuracy_per_mille >= 900, "{trace:?}");
    }
}
