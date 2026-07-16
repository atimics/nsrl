//! Integer gradient accumulation, parameter updates, and transformer backward passes.

use super::*;

pub(super) fn residual_l1_i64(values: &[i64]) -> u64 {
    values.iter().fold(0_u64, |total, value| {
        total.saturating_add(value.unsigned_abs())
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LinearWeightGradientI64 {
    pub(super) input_dim: usize,
    pub(super) output_dim: usize,
    pub(super) sample_count: usize,
    pub(super) accumulators: Vec<i64>,
    pub(super) residuals: Vec<i64>,
}

impl LinearWeightGradientI64 {
    pub(super) fn new(input_dim: usize, output_dim: usize) -> Option<Self> {
        if input_dim == 0 || output_dim == 0 {
            return None;
        }
        let len = input_dim.checked_mul(output_dim)?;
        Some(Self {
            input_dim,
            output_dim,
            sample_count: 0,
            accumulators: vec![0_i64; len],
            residuals: vec![0_i64; len],
        })
    }

    pub(super) fn clear(&mut self) {
        self.sample_count = 0;
        self.accumulators.fill(0);
    }

    pub(super) fn as_train_core_workspace(
        &mut self,
    ) -> nsrl_train_core::LinearWeightGradientI64Workspace<'_> {
        nsrl_train_core::LinearWeightGradientI64Workspace {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            sample_count: self.sample_count,
            accumulators: &mut self.accumulators,
            residuals: &mut self.residuals,
        }
    }

    pub(super) fn residual_l1(&self) -> u64 {
        residual_l1_i64(&self.residuals)
    }
}

pub(super) fn accumulate_linear_weight_gradient_i64_prescaled(
    input: &[i16],
    scaled_grad_output: &[i32],
    gradient: &mut LinearWeightGradientI64,
) -> Result<(), TrainError> {
    let (result, sample_count) = {
        let mut workspace = gradient.as_train_core_workspace();
        let result = nsrl_train_core::accumulate_linear_weight_gradient_i64_prescaled(
            input,
            scaled_grad_output,
            &mut workspace,
        )
        .map_err(|error| {
            train_core_error_to_train_error(error, "linear_weight_gradient_accumulate")
        });
        (result, workspace.sample_count)
    };
    gradient.sample_count = sample_count;
    result
}

pub(super) fn apply_linear_weight_gradient_i64_to_i8(
    gradient: &mut LinearWeightGradientI64,
    weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<LinearWeightUpdateStats, TrainError> {
    let (result, sample_count) = {
        let mut workspace = gradient.as_train_core_workspace();
        let result = nsrl_train_core::apply_linear_weight_gradient_i64_to_i8(
            &mut workspace,
            weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )
        .map_err(|error| train_core_error_to_train_error(error, "linear_weight_gradient_apply"));
        (result, workspace.sample_count)
    };
    gradient.sample_count = sample_count;
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GatedMlpWeightGradientI64 {
    pub(super) down: LinearWeightGradientI64,
    pub(super) up: LinearWeightGradientI64,
    pub(super) gate: LinearWeightGradientI64,
}

impl GatedMlpWeightGradientI64 {
    pub(super) fn new(d_model: usize, hidden_dim: usize) -> Option<Self> {
        Some(Self {
            down: LinearWeightGradientI64::new(hidden_dim, d_model)?,
            up: LinearWeightGradientI64::new(d_model, hidden_dim)?,
            gate: LinearWeightGradientI64::new(d_model, hidden_dim)?,
        })
    }

    pub(super) fn clear(&mut self) {
        self.down.clear();
        self.up.clear();
        self.gate.clear();
    }

    pub(super) fn residual_l1(&self) -> u64 {
        self.down
            .residual_l1()
            .saturating_add(self.up.residual_l1())
            .saturating_add(self.gate.residual_l1())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_gated_mlp_weight_gradient_i64(
    input: &[i16],
    grad_output: &[i16],
    forward_gated: &[i16],
    grad_up: &[i16],
    grad_gate: &[i16],
    params: GatedMlpWeightUpdateParams<'_>,
    gradient: &mut GatedMlpWeightGradientI64,
    scaled_grad_output: &mut [i32],
) -> Result<(), TrainError> {
    if !params.is_valid() {
        return Err(TrainError::InvalidConfig);
    }
    let total = params
        .seq_len
        .checked_mul(params.d_model)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = params
        .seq_len
        .checked_mul(params.hidden_dim)
        .ok_or(TrainError::InvalidConfig)?;
    if input.len() != total
        || grad_output.len() != total
        || forward_gated.len() != hidden_total
        || grad_up.len() != hidden_total
        || grad_gate.len() != hidden_total
        || scaled_grad_output.len() < params.d_model.max(params.hidden_dim)
    {
        return Err(TrainError::InvalidConfig);
    }

    for token in 0..params.seq_len {
        let input_start = token
            .checked_mul(params.d_model)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_input_row"))?;
        let input_end = input_start
            .checked_add(params.d_model)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_input_end"))?;
        let hidden_start = token
            .checked_mul(params.hidden_dim)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_hidden_row"))?;
        let hidden_end = hidden_start
            .checked_add(params.hidden_dim)
            .ok_or(TrainError::CoreRejected("gated_mlp_gradient_hidden_end"))?;

        let grad_row = &grad_output[input_start..input_end];
        if !grad_row.iter().any(|&gradient| gradient != 0) {
            continue;
        }

        linear_backward_prescale_grad_output_i16_i32_checked(
            grad_row,
            params.down_scales,
            &mut scaled_grad_output[..params.d_model],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_down_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &forward_gated[hidden_start..hidden_end],
            &scaled_grad_output[..params.d_model],
            &mut gradient.down,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_up[hidden_start..hidden_end],
            params.up_scales,
            &mut scaled_grad_output[..params.hidden_dim],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_up_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &input[input_start..input_end],
            &scaled_grad_output[..params.hidden_dim],
            &mut gradient.up,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_gate[hidden_start..hidden_end],
            params.gate_scales,
            &mut scaled_grad_output[..params.hidden_dim],
        )
        .ok_or(TrainError::CoreRejected("gated_mlp_gate_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &input[input_start..input_end],
            &scaled_grad_output[..params.hidden_dim],
            &mut gradient.gate,
        )?;
    }

    Ok(())
}

pub(super) fn apply_gated_mlp_weight_gradient_i64_to_i8(
    gradient: &mut GatedMlpWeightGradientI64,
    up_weights: &mut [i8],
    gate_weights: &mut [i8],
    down_weights: &mut [i8],
    learning_rate: i32,
    learning_rate_shift: u8,
    carry_residual: bool,
) -> Result<GatedMlpWeightUpdateStats, TrainError> {
    Ok(GatedMlpWeightUpdateStats {
        down: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.down,
            down_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
        up: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.up,
            up_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
        gate: apply_linear_weight_gradient_i64_to_i8(
            &mut gradient.gate,
            gate_weights,
            learning_rate,
            learning_rate_shift,
            carry_residual,
        )?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MiniTransformerAttentionWeightGradientI64 {
    pub(super) q: LinearWeightGradientI64,
    pub(super) k: LinearWeightGradientI64,
    pub(super) v: LinearWeightGradientI64,
    pub(super) o: LinearWeightGradientI64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MiniTransformerRmsVectorGradientI64 {
    pub(super) sample_count: usize,
    pub(super) accumulators: Vec<i64>,
}

impl MiniTransformerRmsVectorGradientI64 {
    pub(super) fn new() -> Self {
        Self {
            sample_count: 0,
            accumulators: vec![0_i64; MINI_TRANSFORMER_D_MODEL],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MiniTransformerRmsWeightGradientI64 {
    pub(super) attention: MiniTransformerRmsVectorGradientI64,
    pub(super) mlp: MiniTransformerRmsVectorGradientI64,
}

impl MiniTransformerRmsWeightGradientI64 {
    pub(super) fn new() -> Self {
        Self {
            attention: MiniTransformerRmsVectorGradientI64::new(),
            mlp: MiniTransformerRmsVectorGradientI64::new(),
        }
    }
}

impl MiniTransformerAttentionWeightGradientI64 {
    pub(super) fn new(d_model: usize) -> Option<Self> {
        Some(Self {
            q: LinearWeightGradientI64::new(d_model, d_model)?,
            k: LinearWeightGradientI64::new(d_model, d_model)?,
            v: LinearWeightGradientI64::new(d_model, d_model)?,
            o: LinearWeightGradientI64::new(d_model, d_model)?,
        })
    }

    pub(super) fn clear(&mut self) {
        self.q.clear();
        self.k.clear();
        self.v.clear();
        self.o.clear();
    }

    pub(super) fn residual_l1(&self) -> u64 {
        self.q
            .residual_l1()
            .saturating_add(self.k.residual_l1())
            .saturating_add(self.v.residual_l1())
            .saturating_add(self.o.residual_l1())
    }
}

pub(super) fn accumulate_mini_transformer_attention_weight_gradient_i64(
    cache: &MiniTransformerBlockForwardCache,
    grad_attention_output: &[i16],
    grad_q: &[i16],
    grad_k: &[i16],
    grad_v: &[i16],
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    scaled_grad: &mut [i32],
) -> Result<(), TrainError> {
    let seq_len = cache.attention_norm.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || cache.attention_norm.len() != total
        || cache.attention_context.len() != total
        || grad_attention_output.len() != total
        || grad_q.len() != total
        || grad_k.len() != total
        || grad_v.len() != total
        || scaled_grad.len() < MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }

    for token in 0..seq_len {
        let row_start = token
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("attention_gradient_row"))?;
        let row_end = row_start
            .checked_add(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("attention_gradient_row_end"))?;
        let attention_input_row = &cache.attention_norm[row_start..row_end];

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_q[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_q_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.q,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_k[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_k_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.k,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_v[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_v_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            attention_input_row,
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.v,
        )?;

        linear_backward_prescale_grad_output_i16_i32_checked(
            &grad_attention_output[row_start..row_end],
            &MINI_TRANSFORMER_D_MODEL_SCALES,
            &mut scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )
        .ok_or(TrainError::CoreRejected("attention_o_gradient_prescale"))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &cache.attention_context[row_start..row_end],
            &scaled_grad[..MINI_TRANSFORMER_D_MODEL],
            &mut gradient.o,
        )?;
    }

    Ok(())
}

pub(super) fn apply_mini_transformer_attention_weight_gradient_i64_to_i8(
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    model: &mut MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let final_layer_index = model
        .checked_transformer_layers()?
        .checked_sub(1)
        .ok_or(TrainError::InvalidConfig)?;
    apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
        gradient,
        model,
        final_layer_index,
        config,
    )
}

pub(super) fn apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
    gradient: &mut MiniTransformerAttentionWeightGradientI64,
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let transformer_layers = model.checked_transformer_layers()?;
    if layer_index >= transformer_layers {
        return Err(TrainError::InvalidConfig);
    }
    let attention_range = model.attention_weight_range(layer_index)?;
    let q = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.q,
        &mut model.q_weights[attention_range.clone()],
        config.learning_rate,
        config.attention_q_learning_rate_shift,
        true,
    )?;
    let k = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient.k,
        &mut model.k_weights[attention_range.clone()],
        config.learning_rate,
        config.attention_qk_learning_rate_shift,
        true,
    )?;
    let use_vo_oracle = config.attention_vo_oracle && layer_index + 1 == transformer_layers;
    let (v, o) = if use_vo_oracle {
        gradient.v.clear();
        gradient.o.clear();
        (
            empty_linear_weight_update_stats(),
            empty_linear_weight_update_stats(),
        )
    } else {
        (
            apply_linear_weight_gradient_i64_to_i8(
                &mut gradient.v,
                &mut model.v_weights[attention_range.clone()],
                config.learning_rate,
                config.attention_learning_rate_shift,
                true,
            )?,
            apply_linear_weight_gradient_i64_to_i8(
                &mut gradient.o,
                &mut model.o_weights[attention_range],
                config.learning_rate,
                config.attention_learning_rate_shift,
                true,
            )?,
        )
    };

    let mut total = empty_linear_weight_update_stats();
    add_linear_weight_update_stats_checked(&mut total, q)?;
    add_linear_weight_update_stats_checked(&mut total, k)?;
    add_linear_weight_update_stats_checked(&mut total, v)?;
    add_linear_weight_update_stats_checked(&mut total, o)?;

    Ok(MiniTransformerAttentionWeightUpdateStats {
        q,
        k,
        v,
        o,
        gradient_saturation_count: total.gradient_saturation_count,
        zero_delta_count: total.zero_delta_count,
        weight_delta_l1: total.weight_delta_l1,
        grad_embedding_output: Vec::new(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MiniTransformerAttentionVoMatrix {
    Value,
    Output,
}

pub(super) fn mini_transformer_attention_vo_oracle_update_i8_checked(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    step: i32,
) -> Result<(LinearWeightUpdateStats, LinearWeightUpdateStats), TrainError> {
    if starts.is_empty()
        || seq_len == 0
        || step <= 0
        || MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }

    let step = i64::from(step);
    let v = mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
        model,
        tokens,
        starts,
        seq_len,
        MiniTransformerAttentionVoMatrix::Value,
        step,
    )?;
    let o = mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
        model,
        tokens,
        starts,
        seq_len,
        MiniTransformerAttentionVoMatrix::Output,
        step,
    )?;
    Ok((v, o))
}

pub(super) fn mini_transformer_attention_vo_oracle_update_matrix_i8_checked(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    matrix: MiniTransformerAttentionVoMatrix,
    step: i64,
) -> Result<LinearWeightUpdateStats, TrainError> {
    let len = mini_transformer_attention_weight_count()?;

    let mut stats = empty_linear_weight_update_stats();
    let mut current_loss =
        mini_transformer_total_probability_error_q15(tokens, starts, model, seq_len)?;
    for index in 0..len {
        let current = mini_transformer_attention_vo_weight(model, matrix, index)?;
        let mut best_value = current;
        let mut best_loss = current_loss;

        for direction in [1_i64, -1_i64] {
            let candidate_wide = i64::from(current)
                .checked_add(
                    direction
                        .checked_mul(step)
                        .ok_or(TrainError::CoreRejected("attention_vo_oracle_direction"))?,
                )
                .ok_or(TrainError::CoreRejected("attention_vo_oracle_candidate"))?;
            let candidate = saturate_i8(candidate_wide);
            if candidate == current {
                continue;
            }

            mini_transformer_set_attention_vo_weight(model, matrix, index, candidate)?;
            if let Ok(candidate_loss) =
                mini_transformer_total_probability_error_q15(tokens, starts, model, seq_len)
                && candidate_loss < best_loss
            {
                best_loss = candidate_loss;
                best_value = candidate;
            }
        }

        mini_transformer_set_attention_vo_weight(model, matrix, index, best_value)?;
        if best_value == current {
            stats.zero_delta_count =
                stats
                    .zero_delta_count
                    .checked_add(1)
                    .ok_or(TrainError::CoreRejected(
                        "attention_vo_oracle_zero_delta_count",
                    ))?;
        } else {
            let delta = i64::from(best_value) - i64::from(current);
            stats.weight_delta_l1 = stats
                .weight_delta_l1
                .checked_add(delta.unsigned_abs())
                .ok_or(TrainError::CoreRejected("attention_vo_oracle_delta_l1"))?;
            current_loss = best_loss;
        }
    }

    Ok(stats)
}

pub(super) fn mini_transformer_attention_vo_weight(
    model: &MiniTransformerMlpModel,
    matrix: MiniTransformerAttentionVoMatrix,
    index: usize,
) -> Result<i8, TrainError> {
    let range = model.final_attention_weight_range()?;
    let absolute_index = range
        .start
        .checked_add(index)
        .ok_or(TrainError::InvalidConfig)?;
    if absolute_index >= range.end {
        return Err(TrainError::InvalidConfig);
    }
    match matrix {
        MiniTransformerAttentionVoMatrix::Value => model
            .v_weights
            .get(absolute_index)
            .copied()
            .ok_or(TrainError::InvalidConfig),
        MiniTransformerAttentionVoMatrix::Output => model
            .o_weights
            .get(absolute_index)
            .copied()
            .ok_or(TrainError::InvalidConfig),
    }
}

pub(super) fn mini_transformer_set_attention_vo_weight(
    model: &mut MiniTransformerMlpModel,
    matrix: MiniTransformerAttentionVoMatrix,
    index: usize,
    value: i8,
) -> Result<(), TrainError> {
    let range = model.final_attention_weight_range()?;
    let absolute_index = range
        .start
        .checked_add(index)
        .ok_or(TrainError::InvalidConfig)?;
    if absolute_index >= range.end {
        return Err(TrainError::InvalidConfig);
    }
    let slot = match matrix {
        MiniTransformerAttentionVoMatrix::Value => model.v_weights.get_mut(absolute_index),
        MiniTransformerAttentionVoMatrix::Output => model.o_weights.get_mut(absolute_index),
    }
    .ok_or(TrainError::InvalidConfig)?;
    *slot = value;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MiniTransformerEmbeddingGradientI64 {
    pub(super) sample_count: usize,
    pub(super) token_accumulators: Vec<i64>,
    pub(super) position_accumulators: Vec<i64>,
    pub(super) token_residuals: Vec<i64>,
    pub(super) position_residuals: Vec<i64>,
}

impl MiniTransformerEmbeddingGradientI64 {
    pub(super) fn new(context_seq_len: usize) -> Option<Self> {
        let token_len = BYTE_VOCAB.checked_mul(MINI_TRANSFORMER_D_MODEL)?;
        let position_len = context_seq_len.checked_mul(MINI_TRANSFORMER_D_MODEL)?;
        Some(Self {
            sample_count: 0,
            token_accumulators: vec![0_i64; token_len],
            position_accumulators: vec![0_i64; position_len],
            token_residuals: vec![0_i64; token_len],
            position_residuals: vec![0_i64; position_len],
        })
    }

    pub(super) fn clear(&mut self) {
        self.sample_count = 0;
        self.token_accumulators.fill(0);
        self.position_accumulators.fill(0);
    }

    pub(super) fn residual_l1(&self, position_policy: MiniTransformerPositionPolicy) -> u64 {
        let token_l1 = residual_l1_i64(&self.token_residuals);
        if position_policy.uses_position_embeddings() {
            token_l1.saturating_add(residual_l1_i64(&self.position_residuals))
        } else {
            token_l1
        }
    }
}

pub(super) fn accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    position_policy: MiniTransformerPositionPolicy,
    gradient: &mut MiniTransformerEmbeddingGradientI64,
) -> Result<(), TrainError> {
    if context.is_empty()
        || grad_embedding_output_q15.len()
            != context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
        || gradient.token_accumulators.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && gradient.position_accumulators.len()
            < context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    for (position, &token) in context.iter().enumerate() {
        let embedding_row_start = usize::from(token)
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("embedding_gradient_row"))?;
        let position_row_start = position
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("position_embedding_gradient_row"))?;
        let grad_row_start = position
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::CoreRejected("embedding_gradient_grad_row"))?;
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let grad = i64::from(grad_embedding_output_q15[grad_row_start + dim]);
            if grad == 0 {
                continue;
            }
            let index = embedding_row_start
                .checked_add(dim)
                .ok_or(TrainError::CoreRejected("embedding_gradient_index"))?;
            gradient.token_accumulators[index] = gradient.token_accumulators[index]
                .checked_add(grad)
                .ok_or(TrainError::CoreRejected("embedding_gradient_accumulate"))?;
            if position_policy.uses_position_embeddings() {
                let position_index =
                    position_row_start
                        .checked_add(dim)
                        .ok_or(TrainError::CoreRejected(
                            "position_embedding_gradient_index",
                        ))?;
                gradient.position_accumulators[position_index] = gradient.position_accumulators
                    [position_index]
                    .checked_add(grad)
                    .ok_or(TrainError::CoreRejected(
                        "position_embedding_gradient_accumulate",
                    ))?;
            }
        }
    }

    gradient.sample_count = gradient
        .sample_count
        .checked_add(1)
        .ok_or(TrainError::CoreRejected("embedding_gradient_sample_count"))?;
    Ok(())
}

pub(super) fn apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
    gradient: &mut MiniTransformerEmbeddingGradientI64,
    embeddings: &mut [i16],
    position_embeddings: &mut [i16],
    position_policy: MiniTransformerPositionPolicy,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != gradient.token_accumulators.len()
        || embeddings.len() != gradient.token_residuals.len()
        || learning_rate <= 0
        || embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && (position_embeddings.len() != gradient.position_accumulators.len()
            || position_embeddings.len() != gradient.position_residuals.len())
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = empty_softmax_update_stats();
    if gradient.sample_count == 0 {
        return Ok(stats);
    }

    apply_embedding_accumulators_i64_to_i16(
        &gradient.token_accumulators,
        &mut gradient.token_residuals,
        embeddings,
        gradient.sample_count,
        learning_rate,
        embedding_learning_rate_shift,
        &mut stats,
    )?;
    if position_policy.uses_position_embeddings() {
        apply_embedding_accumulators_i64_to_i16(
            &gradient.position_accumulators,
            &mut gradient.position_residuals,
            position_embeddings,
            gradient.sample_count,
            learning_rate,
            embedding_learning_rate_shift,
            &mut stats,
        )?;
    }

    gradient.clear();
    Ok(stats)
}

pub(super) fn apply_embedding_accumulators_i64_to_i16(
    accumulators: &[i64],
    residuals: &mut [i64],
    embeddings: &mut [i16],
    sample_count: usize,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
    stats: &mut SoftmaxUpdateStats,
) -> Result<(), TrainError> {
    if accumulators.len() != residuals.len() || accumulators.len() != embeddings.len() {
        return Err(TrainError::InvalidConfig);
    }

    for ((raw_sum, residual), embedding) in accumulators
        .iter()
        .zip(residuals.iter_mut())
        .zip(embeddings.iter_mut())
    {
        if *raw_sum == 0 && *residual == 0 {
            continue;
        }
        let averaged = round_div_i64(*raw_sum, sample_count)?;
        let product = averaged
            .checked_mul(i64::from(learning_rate))
            .ok_or(TrainError::CoreRejected("embedding_gradient_apply_product"))?;
        let product = product
            .checked_add(*residual)
            .ok_or(TrainError::CoreRejected(
                "embedding_gradient_apply_residual",
            ))?;
        let scaled_update = round_shift_rhu_i64(product, embedding_learning_rate_shift);
        *residual =
            rounded_shift_residual_i64(product, scaled_update, embedding_learning_rate_shift)?;
        let delta = -scaled_update;
        if delta == 0 {
            stats.zero_delta_count = stats.zero_delta_count.saturating_add(1);
        }

        let previous = *embedding;
        let unclamped = i64::from(previous)
            .checked_add(delta)
            .ok_or(TrainError::CoreRejected("embedding_gradient_apply_delta"))?;
        let clamped = saturate_i16(unclamped);
        if i64::from(clamped) != unclamped {
            stats.gradient_saturation_count = stats.gradient_saturation_count.saturating_add(1);
        }
        let applied_delta = i64::from(clamped) - i64::from(previous);
        stats.weight_delta_l1 = stats
            .weight_delta_l1
            .saturating_add(applied_delta.unsigned_abs());
        *embedding = clamped;
    }

    Ok(())
}

pub(super) fn average_mini_transformer_batch_movement(
    base: &MiniTransformerMlpModel,
    model: &mut MiniTransformerMlpModel,
    divisor: usize,
    include_output_weights: bool,
    include_mlp_weights: bool,
    include_attention_weights: bool,
    include_embeddings: bool,
) -> Result<(), TrainError> {
    if divisor == 0 || base.context_seq_len != model.context_seq_len {
        return Err(TrainError::InvalidConfig);
    }

    if include_embeddings {
        average_i16_movement(&base.embeddings, &mut model.embeddings, divisor)?;
        average_i16_movement(
            &base.position_embeddings,
            &mut model.position_embeddings,
            divisor,
        )?;
    }
    if include_attention_weights {
        average_i8_movement(&base.q_weights, &mut model.q_weights, divisor)?;
        average_i8_movement(&base.k_weights, &mut model.k_weights, divisor)?;
        average_i8_movement(&base.v_weights, &mut model.v_weights, divisor)?;
        average_i8_movement(&base.o_weights, &mut model.o_weights, divisor)?;
    }
    if include_mlp_weights {
        average_i8_movement(&base.up_weights, &mut model.up_weights, divisor)?;
        average_i8_movement(&base.gate_weights, &mut model.gate_weights, divisor)?;
        average_i8_movement(&base.down_weights, &mut model.down_weights, divisor)?;
    }
    if include_output_weights {
        average_i8_movement(&base.output_weights, &mut model.output_weights, divisor)?;
    }
    Ok(())
}

pub(super) fn average_i8_movement(
    base: &[i8],
    values: &mut [i8],
    divisor: usize,
) -> Result<(), TrainError> {
    if base.len() != values.len() || divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for (base, value) in base.iter().zip(values.iter_mut()) {
        let movement = i64::from(*value) - i64::from(*base);
        let averaged = round_div_i64(movement, divisor)?;
        *value = saturate_i8(i64::from(*base) + averaged);
    }

    Ok(())
}

pub(super) fn average_i16_movement(
    base: &[i16],
    values: &mut [i16],
    divisor: usize,
) -> Result<(), TrainError> {
    if base.len() != values.len() || divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for (base, value) in base.iter().zip(values.iter_mut()) {
        let movement = i64::from(*value) - i64::from(*base);
        let averaged = round_div_i64(movement, divisor)?;
        *value = saturate_i16(i64::from(*base) + averaged);
    }

    Ok(())
}

pub(super) fn round_div_i64(value: i64, divisor: usize) -> Result<i64, TrainError> {
    if divisor == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let divisor = i64::try_from(divisor).map_err(|_| TrainError::InvalidConfig)?;
    let half = divisor / 2;
    if value >= 0 {
        Ok((value + half) / divisor)
    } else {
        Ok(-((-value + half) / divisor))
    }
}

pub(super) fn rounded_shift_residual_i64(
    value: i64,
    shifted: i64,
    right_shift: u8,
) -> Result<i64, TrainError> {
    if right_shift == 0 {
        return Ok(0);
    }

    let applied = i128::from(shifted)
        .checked_shl(u32::from(right_shift))
        .ok_or(TrainError::CoreRejected("rounded_shift_residual_apply"))?;
    let residual = i128::from(value)
        .checked_sub(applied)
        .ok_or(TrainError::CoreRejected("rounded_shift_residual_subtract"))?;
    i64::try_from(residual).map_err(|_| TrainError::CoreRejected("rounded_shift_residual_range"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mini_transformer_validate_guard_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    epoch: usize,
    window_index: usize,
    epochs: usize,
) -> Result<(), TrainError> {
    if starts.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let len = starts.len();
    let candidates = [
        0,
        len / 4,
        len / 2,
        (len * 3) / 4,
        len - 1,
        window_index.saturating_sub(1),
        window_index,
        (window_index + 1).min(len - 1),
        if window_index + 1 < len {
            window_index + 1
        } else if epoch + 1 < epochs {
            0
        } else {
            window_index
        },
    ];

    let mut seen = [usize::MAX; 9];
    let mut seen_len = 0_usize;
    for &index in candidates.iter() {
        if index >= len || seen[..seen_len].contains(&index) {
            continue;
        }
        seen[seen_len] = index;
        seen_len += 1;

        let start = starts[index];
        mini_transformer_forward_for_attention_and_position(
            model,
            &tokens[start..start + seq_len],
            attention_kind,
            position_policy,
        )?;
    }

    Ok(())
}

pub(super) fn mini_transformer_validate_batch_windows(
    model: &MiniTransformerMlpModel,
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<(), TrainError> {
    if starts.is_empty() || seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    for &start in starts {
        mini_transformer_forward_for_attention_and_position(
            model,
            &tokens[start..start + seq_len],
            attention_kind,
            position_policy,
        )?;
    }

    Ok(())
}

pub(super) fn mini_transformer_block_backward_update_i8_checked(
    cache: &MiniTransformerBlockForwardCache,
    grad_block_output: &[i16],
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
) -> Result<MiniTransformerBlockBackwardUpdate, TrainError> {
    let seq_len = cache.block_output.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || grad_block_output.len() != total
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
        || cache.attention_output.len() != total
        || cache.attention_residual.len() != total
        || cache.mlp_norm.len() != total
        || cache.mlp_up.len() != hidden_total
        || cache.mlp_gate.len() != hidden_total
        || cache.mlp_gated.len() != hidden_total
        || cache.mlp_output.len() != total
        || cache.block_output.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    workspace.grad_mlp_output[..total].copy_from_slice(grad_block_output);

    let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
        &workspace.grad_mlp_output[..total],
        mini_transformer_mlp_params_for_layer(model, layer_index, seq_len)?,
        &cache.mlp_up,
        &cache.mlp_gate,
        GatedMlpBackwardScales {
            down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
            gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
        },
        GatedMlpBackwardWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_input_grad_gated,
            grad_up: &mut workspace.mlp_input_grad_up,
            grad_gate: &mut workspace.mlp_input_grad_gate,
            grad_up_input: &mut workspace.mlp_input_grad_up_input,
            grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
        },
        &mut workspace.grad_mlp_input,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_backward_input",
    ))?;

    let gradient_residual_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_mlp_output[..total],
        &workspace.grad_mlp_input[..total],
        &mut workspace.grad_attention_output[..total],
    )?;

    let up_or_gate_range = model.mlp_up_or_gate_weight_range(layer_index)?;
    let down_range = model.mlp_down_weight_range(layer_index)?;
    let mlp_update = gated_mlp_backward_weight_update_i8_checked(
        &cache.mlp_norm,
        &workspace.grad_mlp_output[..total],
        &cache.mlp_up,
        &cache.mlp_gate,
        &cache.mlp_gated,
        &mut model.up_weights[up_or_gate_range.clone()],
        &mut model.gate_weights[up_or_gate_range],
        &mut model.down_weights[down_range],
        GatedMlpWeightUpdateParams {
            up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            seq_len,
            d_model: MINI_TRANSFORMER_D_MODEL,
            hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.mlp_learning_rate_shift,
        },
        GatedMlpWeightUpdateWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_update_grad_gated,
            grad_up: &mut workspace.mlp_update_grad_up,
            grad_gate: &mut workspace.mlp_update_grad_gate,
        },
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_update",
    ))?;

    let attention_update = mini_transformer_attention_update_i8_checked(
        cache,
        model,
        layer_index,
        config,
        workspace,
        None,
    )?;

    let mut grad_input = vec![0_i16; total];
    let input_gradient_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_attention_output[..total],
        &workspace.grad_attention_norm_input[..total],
        &mut grad_input,
    )?;

    Ok(MiniTransformerBlockBackwardUpdate {
        mlp_update,
        attention_update,
        mlp_input_saturation_count,
        gradient_residual_saturation_count,
        input_gradient_saturation_count,
        grad_input,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mini_transformer_block_backward_accumulate_i64_checked(
    cache: &MiniTransformerBlockForwardCache,
    grad_block_output: &[i16],
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
    mlp_gradient: &mut GatedMlpWeightGradientI64,
    attention_gradient: &mut MiniTransformerAttentionWeightGradientI64,
    rms_gradient: &mut MiniTransformerRmsWeightGradientI64,
) -> Result<MiniTransformerBlockBackwardAccumulation, TrainError> {
    let seq_len = cache.block_output.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || grad_block_output.len() != total
        || cache.block_input.len() != total
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
        || cache.attention_output.len() != total
        || cache.attention_residual.len() != total
        || cache.mlp_norm.len() != total
        || cache.mlp_up.len() != hidden_total
        || cache.mlp_gate.len() != hidden_total
        || cache.mlp_gated.len() != hidden_total
        || cache.mlp_output.len() != total
        || cache.block_output.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let rms_weights = if model.rms_norm_enabled() {
        let range = model.rms_weight_range(layer_index)?;
        Some((
            model.attention_rms_weights[range.clone()].to_vec(),
            model.mlp_rms_weights[range].to_vec(),
        ))
    } else {
        None
    };
    workspace.grad_mlp_output[..total].copy_from_slice(grad_block_output);

    let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
        &workspace.grad_mlp_output[..total],
        mini_transformer_mlp_params_for_layer(model, layer_index, seq_len)?,
        &cache.mlp_up,
        &cache.mlp_gate,
        GatedMlpBackwardScales {
            down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
            gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
        },
        GatedMlpBackwardWorkspace {
            scaled_grad_output: &mut workspace.mlp_scaled_grad,
            grad_gated: &mut workspace.mlp_input_grad_gated,
            grad_up: &mut workspace.mlp_input_grad_up,
            grad_gate: &mut workspace.mlp_input_grad_gate,
            grad_up_input: &mut workspace.mlp_input_grad_up_input,
            grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
        },
        &mut workspace.grad_mlp_input,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_block_mlp_backward_input",
    ))?;

    let mut grad_mlp_residual = vec![0_i16; total];
    let mlp_rms_backward_saturation_count = if let Some((_, mlp_weights)) = &rms_weights {
        mini_transformer_rms_norm_backward_rows(
            &cache.attention_residual,
            mlp_weights,
            &workspace.grad_mlp_input[..total],
            &mut grad_mlp_residual,
            &mut rms_gradient.mlp,
        )?
    } else {
        grad_mlp_residual.copy_from_slice(&workspace.grad_mlp_input[..total]);
        0
    };
    let gradient_residual_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_mlp_output[..total],
        &grad_mlp_residual,
        &mut workspace.grad_attention_output[..total],
    )?;

    accumulate_gated_mlp_weight_gradient_i64(
        &cache.mlp_norm,
        &workspace.grad_mlp_output[..total],
        &cache.mlp_gated,
        &workspace.mlp_input_grad_up,
        &workspace.mlp_input_grad_gate,
        GatedMlpWeightUpdateParams {
            up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
            seq_len,
            d_model: MINI_TRANSFORMER_D_MODEL,
            hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            learning_rate: config.learning_rate,
            learning_rate_shift: config.mlp_learning_rate_shift,
        },
        mlp_gradient,
        &mut workspace.mlp_scaled_grad,
    )?;

    let attention_update = mini_transformer_attention_update_i8_checked(
        cache,
        model,
        layer_index,
        config,
        workspace,
        Some(attention_gradient),
    )?;

    let mut grad_attention_input = vec![0_i16; total];
    let attention_rms_backward_saturation_count = if let Some((attention_weights, _)) = &rms_weights
    {
        mini_transformer_rms_norm_backward_rows(
            &cache.block_input,
            attention_weights,
            &workspace.grad_attention_norm_input[..total],
            &mut grad_attention_input,
            &mut rms_gradient.attention,
        )?
    } else {
        grad_attention_input.copy_from_slice(&workspace.grad_attention_norm_input[..total]);
        0
    };
    let mut grad_input = vec![0_i16; total];
    let input_gradient_saturation_count = add_i16_residual_rows_checked(
        &workspace.grad_attention_output[..total],
        &grad_attention_input,
        &mut grad_input,
    )?;

    Ok(MiniTransformerBlockBackwardAccumulation {
        mlp_input_saturation_count: mlp_input_saturation_count
            .saturating_add(mlp_rms_backward_saturation_count),
        attention_gradient_saturation_count: attention_update.gradient_saturation_count,
        gradient_residual_saturation_count,
        input_gradient_saturation_count: input_gradient_saturation_count
            .saturating_add(attention_rms_backward_saturation_count),
        grad_input,
    })
}

pub(super) fn mini_transformer_attention_update_i8_checked(
    cache: &MiniTransformerBlockForwardCache,
    model: &mut MiniTransformerMlpModel,
    layer_index: usize,
    config: MiniTransformerMlpTrainConfig,
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
    attention_gradient: Option<&mut MiniTransformerAttentionWeightGradientI64>,
) -> Result<MiniTransformerAttentionWeightUpdateStats, TrainError> {
    let seq_len = cache.attention_norm.len() / MINI_TRANSFORMER_D_MODEL;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || cache.attention_norm.len() != total
        || cache.attention_q.len() != total
        || cache.attention_k.len() != total
        || cache.attention_v.len() != total
        || cache.attention_context.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let expected_probability_count = mini_transformer_attention_probability_count(seq_len)?;
    match config.attention_kind {
        MiniTransformerAttentionKind::Base2Softmax => {
            if cache.attention_probabilities_q15.len() != expected_probability_count {
                return Err(TrainError::InvalidConfig);
            }
        }
        MiniTransformerAttentionKind::Linear => {
            if !cache.attention_probabilities_q15.is_empty() {
                return Err(TrainError::InvalidConfig);
            }
        }
        MiniTransformerAttentionKind::LinearStreamingNope
        | MiniTransformerAttentionKind::LinearStreamingTttNope => {
            return Err(TrainError::InvalidConfig);
        }
    }
    let attention_range = model.attention_weight_range(layer_index)?;

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_output[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.o_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut workspace.grad_attention_context[row_start..row_end],
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_o_backward_input",
        ))?;
    }

    match config.attention_kind {
        MiniTransformerAttentionKind::Base2Softmax => {
            let grad_v = mini_transformer_attention_v_gradient_q15(
                seq_len,
                &cache.attention_probabilities_q15,
                &workspace.grad_attention_context,
            )?;
            let grad_probabilities = mini_transformer_attention_probability_gradient_q15(
                seq_len,
                &cache.attention_v,
                &workspace.grad_attention_context,
            )?;
            let grad_logits = mini_transformer_attention_logit_gradient_q15(
                seq_len,
                &cache.attention_probabilities_q15,
                &grad_probabilities,
            )?;
            let (grad_q, grad_k) = mini_transformer_attention_q_k_gradients_q15(
                seq_len,
                &cache.attention_q,
                &cache.attention_k,
                &grad_logits,
            )?;
            workspace.grad_attention_q[..total].copy_from_slice(&grad_q);
            workspace.grad_attention_k[..total].copy_from_slice(&grad_k);
            workspace.grad_attention_v[..total].copy_from_slice(&grad_v);
        }
        MiniTransformerAttentionKind::Linear => {
            mini_transformer_linear_attention_qkv_gradients_q15_workspace(
                seq_len,
                &cache.attention_q,
                &cache.attention_k,
                &cache.attention_v,
                workspace,
            )?;
        }
        MiniTransformerAttentionKind::LinearStreamingNope
        | MiniTransformerAttentionKind::LinearStreamingTttNope => {
            return Err(TrainError::InvalidConfig);
        }
    };
    let mut input_gradient_saturation_count = 0_usize;

    for token in 0..seq_len {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
        let mut grad_q_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_k_input = [0_i16; MINI_TRANSFORMER_D_MODEL];
        let mut grad_v_input = [0_i16; MINI_TRANSFORMER_D_MODEL];

        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_q[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.q_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_q_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_q_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_k[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.k_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_k_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_k_backward_input",
        ))?;
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &workspace.grad_attention_v[row_start..row_end],
            LinearBackwardInputI16I8Params {
                weights: &model.v_weights[attention_range.clone()],
                forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: MINI_TRANSFORMER_D_MODEL,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.attention_scaled_grad
                    [..MINI_TRANSFORMER_D_MODEL],
            },
            &mut grad_v_input,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_attention_v_backward_input",
        ))?;

        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let wide = i64::from(grad_q_input[dim])
                + i64::from(grad_k_input[dim])
                + i64::from(grad_v_input[dim]);
            let scaled = round_shift_rhu_i64(wide, MINI_TRANSFORMER_EMBEDDING_GRAD_FANIN_SHIFT);
            if scaled < i64::from(i16::MIN) || scaled > i64::from(i16::MAX) {
                input_gradient_saturation_count = input_gradient_saturation_count.saturating_add(1);
            }
            workspace.grad_attention_norm_input[row_start + dim] = saturate_i16(scaled);
        }
    }

    let mut total_stats = empty_linear_weight_update_stats();
    let mut q_total = empty_linear_weight_update_stats();
    let mut k_total = empty_linear_weight_update_stats();
    let mut v_total = empty_linear_weight_update_stats();
    let mut o_total = empty_linear_weight_update_stats();
    if let Some(attention_gradient) = attention_gradient {
        accumulate_mini_transformer_attention_weight_gradient_i64(
            cache,
            &workspace.grad_attention_output,
            &workspace.grad_attention_q,
            &workspace.grad_attention_k,
            &workspace.grad_attention_v,
            attention_gradient,
            &mut workspace.attention_scaled_grad[..MINI_TRANSFORMER_D_MODEL],
        )?;
    } else {
        for token in 0..seq_len {
            let row_start = token * MINI_TRANSFORMER_D_MODEL;
            let row_end = row_start + MINI_TRANSFORMER_D_MODEL;
            let q_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_q[row_start..row_end],
                &mut model.q_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_q_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_q_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, q_stats)?;
            add_linear_weight_update_stats_checked(&mut q_total, q_stats)?;

            let k_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_k[row_start..row_end],
                &mut model.k_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_qk_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_k_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, k_stats)?;
            add_linear_weight_update_stats_checked(&mut k_total, k_stats)?;

            let v_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_norm[row_start..row_end],
                &workspace.grad_attention_v[row_start..row_end],
                &mut model.v_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_v_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, v_stats)?;
            add_linear_weight_update_stats_checked(&mut v_total, v_stats)?;

            let o_stats = linear_backward_weight_update_i8_checked(
                &cache.attention_context[row_start..row_end],
                &workspace.grad_attention_output[row_start..row_end],
                &mut model.o_weights[attention_range.clone()],
                LinearBackwardWeightUpdateI8Params {
                    forward_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    input_dim: MINI_TRANSFORMER_D_MODEL,
                    output_dim: MINI_TRANSFORMER_D_MODEL,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.attention_learning_rate_shift,
                },
                LinearBackwardWeightUpdateWorkspace {
                    scaled_grad_output: &mut workspace.attention_scaled_grad
                        [..MINI_TRANSFORMER_D_MODEL],
                },
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_o_update",
            ))?;
            add_linear_weight_update_stats_checked(&mut total_stats, o_stats)?;
            add_linear_weight_update_stats_checked(&mut o_total, o_stats)?;
        }
    }

    Ok(MiniTransformerAttentionWeightUpdateStats {
        q: q_total,
        k: k_total,
        v: v_total,
        o: o_total,
        gradient_saturation_count: total_stats
            .gradient_saturation_count
            .saturating_add(input_gradient_saturation_count),
        zero_delta_count: total_stats.zero_delta_count,
        weight_delta_l1: total_stats.weight_delta_l1,
        grad_embedding_output: Vec::new(),
    })
}

#[cfg(test)]
pub(super) type MiniTransformerLinearAttentionQkvGradients = (Vec<i16>, Vec<i16>, Vec<i16>);

pub(super) fn mini_transformer_linear_attention_qkv_gradients_q15_workspace(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    workspace: &mut MiniTransformerHostTrainCoreWorkspaceBuffers,
) -> Result<(), TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0 || q.len() != total || k.len() != total || v.len() != total {
        return Err(TrainError::InvalidConfig);
    }
    workspace.validate_host_training_step_shape(seq_len)?;
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainError::InvalidConfig)?;
    let state_len = MINI_TRANSFORMER_HEADS
        .checked_mul(head_state_len)
        .ok_or(TrainError::InvalidConfig)?;
    workspace.linear_grad_q_acc[..total].fill(0);
    workspace.linear_grad_k_acc[..total].fill(0);
    workspace.linear_grad_v_acc[..total].fill(0);

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_head_start = head
            .checked_mul(seq_len)
            .and_then(|value| value.checked_mul(head_state_len))
            .ok_or(TrainError::InvalidConfig)?;
        let denom_head_start = head.checked_mul(seq_len).ok_or(TrainError::InvalidConfig)?;
        workspace.linear_grad_state_q15[..head_state_len].fill(0);
        let state_start = head
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let state_end = state_start
            .checked_add(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_start = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_end = key_sum_start
            .checked_add(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        workspace.attention_state_kv[state_start..state_end].fill(0);
        workspace.attention_key_sums[key_sum_start..key_sum_end].fill(0);

        for token in 0..seq_len {
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let state = &mut workspace.attention_state_kv[state_start..state_end];
            let key_sums = &mut workspace.attention_key_sums[key_sum_start..key_sum_end];

            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                key_sums[key_index] =
                    key_sums[key_index]
                        .checked_add(phi_key)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_key_sum",
                        ))?;
                let state_row_start = key_index
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_index, &value_value) in value.iter().enumerate() {
                    let product = phi_key.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_linear_attention_state_product"),
                    )?;
                    let state_index = state_row_start
                        .checked_add(value_index)
                        .ok_or(TrainError::InvalidConfig)?;
                    state[state_index] =
                        state[state_index]
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_state_accumulate",
                            ))?;
                }
            }

            let query = &q[row_start..row_end];
            let mut denominator = 0_i64;
            for (&query_value, &key_sum) in query.iter().zip(key_sums.iter()) {
                let product = mini_transformer_linear_attention_phi_i64(query_value)
                    .checked_mul(key_sum)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator_product",
                    ))?;
                denominator = denominator
                    .checked_add(product)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator",
                    ))?;
            }
            if denominator <= 0 {
                return Err(TrainError::CoreRejected(
                    "mini_transformer_linear_attention_nonpositive_denominator",
                ));
            }
            workspace.linear_denominators[denom_head_start + token] = denominator;
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainError::InvalidConfig)?,
                )
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            workspace.linear_prefix_states[snapshot_start..snapshot_end].copy_from_slice(state);
        }

        for query_index in 0..seq_len {
            let token = seq_len - 1 - query_index;
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let query = &q[row_start..row_end];
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let grad_row = &workspace.grad_attention_context[row_start..row_end];
            let denominator = workspace.linear_denominators[denom_head_start + token];
            let snapshot_start = prefix_head_start
                .checked_add(
                    token
                        .checked_mul(head_state_len)
                        .ok_or(TrainError::InvalidConfig)?,
                )
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let prefix_state = &workspace.linear_prefix_states[snapshot_start..snapshot_end];

            for key_dim in 0..head_dim {
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_q_numerator = 0_i64;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let product = i64::from(grad_value)
                        .checked_mul(prefix_state[state_index])
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_q_gradient_product",
                        ))?;
                    grad_q_numerator =
                        grad_q_numerator
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_q_gradient_accumulate",
                            ))?;
                }
                let target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                workspace.linear_grad_q_acc[target] = workspace.linear_grad_q_acc[target]
                    .checked_add(round_ratio_i64(grad_q_numerator, denominator)?)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_q_gradient_accumulate",
                    ))?;
            }

            for (key_dim, &query_value) in query.iter().enumerate() {
                let phi_query = mini_transformer_linear_attention_phi_i64(query_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let product = i64::from(grad_value)
                        .checked_mul(phi_query)
                        .and_then(|value| value.checked_mul(1_i64 << Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_product",
                        ))?;
                    let state_grad = round_ratio_i64(product, denominator)?;
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.linear_grad_state_q15[state_index] = workspace.linear_grad_state_q15
                        [state_index]
                        .checked_add(state_grad)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_accumulate",
                        ))?;
                }
            }

            for (key_dim, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_key_value = 0_i64;
                for (value_dim, &value_value) in value.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let state_grad = workspace.linear_grad_state_q15[state_index];
                    let grad_v_product =
                        state_grad
                            .checked_mul(phi_key)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_v_gradient_product",
                            ))?;
                    let v_target = row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.linear_grad_v_acc[v_target] = workspace.linear_grad_v_acc[v_target]
                        .checked_add(round_shift_rhu_i64(grad_v_product, Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_v_gradient_accumulate",
                        ))?;

                    let grad_k_product = state_grad.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_product",
                        ),
                    )?;
                    grad_key_value = grad_key_value
                        .checked_add(round_shift_rhu_i64(grad_k_product, Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_accumulate",
                        ))?;
                }
                let k_target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                workspace.linear_grad_k_acc[k_target] = workspace.linear_grad_k_acc[k_target]
                    .checked_add(grad_key_value)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_k_gradient_accumulate",
                    ))?;
            }
        }

        debug_assert!(state_len <= workspace.attention_state_kv.len());
    }

    for index in 0..total {
        workspace.grad_attention_q[index] = saturate_i16(workspace.linear_grad_q_acc[index]);
        workspace.grad_attention_k[index] = saturate_i16(workspace.linear_grad_k_acc[index]);
        workspace.grad_attention_v[index] = saturate_i16(workspace.linear_grad_v_acc[index]);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn mini_transformer_linear_attention_qkv_gradients_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    grad_context: &[i16],
) -> Result<MiniTransformerLinearAttentionQkvGradients, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0
        || q.len() != total
        || k.len() != total
        || v.len() != total
        || grad_context.len() != total
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_q_acc = vec![0_i64; total];
    let mut grad_k_acc = vec![0_i64; total];
    let mut grad_v_acc = vec![0_i64; total];
    let head_state_len = head_dim
        .checked_mul(head_dim)
        .ok_or(TrainError::InvalidConfig)?;

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let mut prefix_states = vec![
            0_i64;
            seq_len
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?
        ];
        let mut denominators = vec![0_i64; seq_len];
        let mut state = vec![0_i64; head_state_len];
        let mut key_sums = vec![0_i64; head_dim];

        for (token, denominator_slot) in denominators.iter_mut().enumerate().take(seq_len) {
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];

            for (key_index, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                key_sums[key_index] =
                    key_sums[key_index]
                        .checked_add(phi_key)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_key_sum",
                        ))?;
                let state_row_start = key_index
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_index, &value_value) in value.iter().enumerate() {
                    let product = phi_key.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_linear_attention_state_product"),
                    )?;
                    let state_index = state_row_start
                        .checked_add(value_index)
                        .ok_or(TrainError::InvalidConfig)?;
                    state[state_index] =
                        state[state_index]
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_state_accumulate",
                            ))?;
                }
            }

            let query = &q[row_start..row_end];
            let mut denominator = 0_i64;
            for (&query_value, &key_sum) in query.iter().zip(key_sums.iter()) {
                let product = mini_transformer_linear_attention_phi_i64(query_value)
                    .checked_mul(key_sum)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator_product",
                    ))?;
                denominator = denominator
                    .checked_add(product)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_linear_attention_denominator",
                    ))?;
            }
            if denominator <= 0 {
                return Err(TrainError::CoreRejected(
                    "mini_transformer_linear_attention_nonpositive_denominator",
                ));
            }

            *denominator_slot = denominator;
            let snapshot_start = token
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            prefix_states[snapshot_start..snapshot_end].copy_from_slice(&state);
        }

        let mut grad_state_q15 = vec![0_i64; head_state_len];
        for query_index in 0..seq_len {
            let token = seq_len - 1 - query_index;
            let row_start = token
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let query = &q[row_start..row_end];
            let key = &k[row_start..row_end];
            let value = &v[row_start..row_end];
            let grad_row = &grad_context[row_start..row_end];
            let denominator = denominators[token];
            let snapshot_start = token
                .checked_mul(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let snapshot_end = snapshot_start
                .checked_add(head_state_len)
                .ok_or(TrainError::InvalidConfig)?;
            let prefix_state = &prefix_states[snapshot_start..snapshot_end];

            for key_dim in 0..head_dim {
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_q_numerator = 0_i64;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let product = i64::from(grad_value)
                        .checked_mul(prefix_state[state_index])
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_q_gradient_product",
                        ))?;
                    grad_q_numerator =
                        grad_q_numerator
                            .checked_add(product)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_q_gradient_accumulate",
                            ))?;
                }
                let grad_q_value = round_ratio_i64(grad_q_numerator, denominator)?;
                let target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                grad_q_acc[target] = grad_q_acc[target].checked_add(grad_q_value).ok_or(
                    TrainError::CoreRejected(
                        "mini_transformer_linear_attention_q_gradient_accumulate",
                    ),
                )?;
            }

            for (key_dim, &query_value) in query.iter().enumerate() {
                let phi_query = mini_transformer_linear_attention_phi_i64(query_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                for (value_dim, &grad_value) in grad_row.iter().enumerate() {
                    let product = i64::from(grad_value)
                        .checked_mul(phi_query)
                        .and_then(|value| value.checked_mul(1_i64 << Q15_SHIFT))
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_product",
                        ))?;
                    let state_grad = round_ratio_i64(product, denominator)?;
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    grad_state_q15[state_index] = grad_state_q15[state_index]
                        .checked_add(state_grad)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_linear_attention_state_gradient_accumulate",
                        ))?;
                }
            }

            for (key_dim, &key_value) in key.iter().enumerate() {
                let phi_key = mini_transformer_linear_attention_phi_i64(key_value);
                let state_row_start = key_dim
                    .checked_mul(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                let mut grad_key_value = 0_i64;
                for (value_dim, &value_value) in value.iter().enumerate() {
                    let state_index = state_row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    let state_grad = grad_state_q15[state_index];
                    let grad_v_product =
                        state_grad
                            .checked_mul(phi_key)
                            .ok_or(TrainError::CoreRejected(
                                "mini_transformer_linear_attention_v_gradient_product",
                            ))?;
                    let grad_v_value = round_shift_rhu_i64(grad_v_product, Q15_SHIFT);
                    let v_target = row_start
                        .checked_add(value_dim)
                        .ok_or(TrainError::InvalidConfig)?;
                    grad_v_acc[v_target] = grad_v_acc[v_target].checked_add(grad_v_value).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_v_gradient_accumulate",
                        ),
                    )?;

                    let grad_k_product = state_grad.checked_mul(i64::from(value_value)).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_product",
                        ),
                    )?;
                    let grad_k_value = round_shift_rhu_i64(grad_k_product, Q15_SHIFT);
                    grad_key_value = grad_key_value.checked_add(grad_k_value).ok_or(
                        TrainError::CoreRejected(
                            "mini_transformer_linear_attention_k_gradient_accumulate",
                        ),
                    )?;
                }
                let k_target = row_start
                    .checked_add(key_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                grad_k_acc[k_target] = grad_k_acc[k_target].checked_add(grad_key_value).ok_or(
                    TrainError::CoreRejected(
                        "mini_transformer_linear_attention_k_gradient_accumulate",
                    ),
                )?;
            }
        }
    }

    let mut grad_q = vec![0_i16; total];
    let mut grad_k = vec![0_i16; total];
    let mut grad_v = vec![0_i16; total];
    for index in 0..total {
        grad_q[index] = saturate_i16(grad_q_acc[index]);
        grad_k[index] = saturate_i16(grad_k_acc[index]);
        grad_v[index] = saturate_i16(grad_v_acc[index]);
    }

    Ok((grad_q, grad_k, grad_v))
}

pub(super) fn mini_transformer_linear_attention_phi_i64(value: i16) -> i64 {
    i64::from(value) + 32769
}

pub(super) fn round_ratio_i64(numerator: i64, denominator: i64) -> Result<i64, TrainError> {
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    if numerator >= 0 {
        numerator
            .checked_add(half)
            .map(|value| value / denominator)
            .ok_or(TrainError::CoreRejected("round_ratio_positive"))
    } else {
        numerator
            .checked_neg()
            .and_then(|value| value.checked_add(half))
            .map(|value| -(value / denominator))
            .ok_or(TrainError::CoreRejected("round_ratio_negative"))
    }
}

pub(super) fn mini_transformer_attention_probability_gradient_q15(
    seq_len: usize,
    values: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if seq_len == 0 || values.len() != total || grad_context.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_probabilities =
        vec![0_i16; mini_transformer_attention_probability_count(seq_len)?];
    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for query_index in 0..seq_len {
            let prob_row =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            for key_index in 0..seq_len {
                if key_index > query_index {
                    continue;
                }

                let query_start = query_index
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_add(head_offset))
                    .ok_or(TrainError::InvalidConfig)?;
                let key_start = key_index
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_add(head_offset))
                    .ok_or(TrainError::InvalidConfig)?;
                let mut acc = 0_i64;
                for dim in 0..head_dim {
                    let grad = grad_context[query_start + dim];
                    let value = values[key_start + dim];
                    let product = i64::from(grad).checked_mul(i64::from(value)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_probability_gradient"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_probability_gradient_accumulate",
                    ))?;
                }

                grad_probabilities[prob_row + key_index] =
                    saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            }
        }
    }

    Ok(grad_probabilities)
}

pub(super) fn mini_transformer_attention_logit_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_probabilities_q15: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0
        || probabilities_q15.len() != expected
        || grad_probabilities_q15.len() != expected
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_logits = vec![0_i16; expected];
    for head in 0..MINI_TRANSFORMER_HEADS {
        for query_index in 0..seq_len {
            let row_start =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            let row_end = row_start
                .checked_add(seq_len)
                .ok_or(TrainError::InvalidConfig)?;
            let probabilities = &probabilities_q15[row_start..row_end];
            let grad_probabilities = &grad_probabilities_q15[row_start..row_end];

            let mut weighted_grad = 0_i64;
            for key_index in 0..=query_index {
                let probability = probabilities[key_index];
                if probability < 0 {
                    return Err(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_negative_probability",
                    ));
                }
                let product = i64::from(grad_probabilities[key_index])
                    .checked_mul(i64::from(probability))
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_weighted_product",
                    ))?;
                weighted_grad =
                    weighted_grad
                        .checked_add(product)
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_attention_logit_weighted_accumulate",
                        ))?;
            }
            let weighted_grad_q15 = round_shift_rhu_i64(weighted_grad, Q15_SHIFT);

            for key_index in 0..=query_index {
                let probability = probabilities[key_index];
                let centered = i64::from(grad_probabilities[key_index])
                    .checked_sub(weighted_grad_q15)
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_center",
                    ))?;
                let product = i64::from(probability)
                    .checked_mul(centered)
                    .and_then(|value| value.checked_mul(i64::from(BASE2_SOFTMAX_LN2_Q15)))
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_logit_gradient",
                    ))?;
                grad_logits[row_start + key_index] = saturate_i16(round_shift_rhu_i64(
                    product,
                    Q15_SHIFT.checked_mul(2).ok_or(TrainError::InvalidConfig)?,
                ));
            }
        }
    }

    Ok(grad_logits)
}

pub(super) fn mini_transformer_attention_q_k_gradients_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
    grad_logits_q15: &[i16],
) -> Result<(Vec<i16>, Vec<i16>), TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0 || q.len() != total || k.len() != total || grad_logits_q15.len() != expected {
        return Err(TrainError::InvalidConfig);
    }

    let sqrt_shift = sqrt_power_of_four_shift(head_dim).ok_or(TrainError::CoreRejected(
        "mini_transformer_attention_qk_sqrt_shift",
    ))?;
    let mut grad_q = vec![0_i16; total];
    let mut grad_k = vec![0_i16; total];

    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for query_index in 0..seq_len {
            let prob_row =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            let query_start = query_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for key_index in 0..=query_index {
                    let grad_logit = grad_logits_q15[prob_row + key_index];
                    if grad_logit == 0 {
                        continue;
                    }
                    let key_start = key_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let key = k[key_start + dim];
                    let product = i64::from(grad_logit).checked_mul(i64::from(key)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_q_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_q_gradient_accumulate",
                    ))?;
                }
                grad_q[query_start + dim] = saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
            }
        }

        for key_index in 0..seq_len {
            let key_start = key_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for query_index in key_index..seq_len {
                    let prob_row = mini_transformer_attention_probability_row_start(
                        head,
                        query_index,
                        seq_len,
                    )?;
                    let grad_logit = grad_logits_q15[prob_row + key_index];
                    if grad_logit == 0 {
                        continue;
                    }
                    let query_start = query_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let query = q[query_start + dim];
                    let product = i64::from(grad_logit).checked_mul(i64::from(query)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_k_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_k_gradient_accumulate",
                    ))?;
                }
                grad_k[key_start + dim] = saturate_i16(round_shift_rhu_i64(acc, sqrt_shift));
            }
        }
    }

    Ok((grad_q, grad_k))
}

pub(super) fn mini_transformer_attention_v_gradient_q15(
    seq_len: usize,
    probabilities_q15: &[i16],
    grad_context: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let expected = mini_transformer_attention_probability_count(seq_len)?;
    if seq_len == 0 || probabilities_q15.len() != expected || grad_context.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut grad_v = vec![0_i16; total];
    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for key_index in 0..seq_len {
            let key_start = key_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            for dim in 0..head_dim {
                let mut acc = 0_i64;
                for query_index in 0..seq_len {
                    let prob_row = mini_transformer_attention_probability_row_start(
                        head,
                        query_index,
                        seq_len,
                    )?;
                    let probability = probabilities_q15[prob_row + key_index];
                    if probability < 0 {
                        return Err(TrainError::CoreRejected(
                            "mini_transformer_attention_v_negative_probability",
                        ));
                    }
                    if probability == 0 {
                        continue;
                    }

                    let query_start = query_index
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .and_then(|value| value.checked_add(head_offset))
                        .ok_or(TrainError::InvalidConfig)?;
                    let grad = grad_context[query_start + dim];
                    let product = i64::from(probability).checked_mul(i64::from(grad)).ok_or(
                        TrainError::CoreRejected("mini_transformer_attention_v_gradient_product"),
                    )?;
                    acc = acc.checked_add(product).ok_or(TrainError::CoreRejected(
                        "mini_transformer_attention_v_gradient_accumulate",
                    ))?;
                }

                grad_v[key_start + dim] = saturate_i16(round_shift_rhu_i64(acc, Q15_SHIFT));
            }
        }
    }

    Ok(grad_v)
}

pub(super) fn empty_linear_weight_update_stats() -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    }
}

pub(super) fn empty_softmax_update_stats() -> SoftmaxUpdateStats {
    SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    }
}

pub(super) fn empty_gated_mlp_weight_update_stats() -> GatedMlpWeightUpdateStats {
    GatedMlpWeightUpdateStats {
        down: empty_linear_weight_update_stats(),
        up: empty_linear_weight_update_stats(),
        gate: empty_linear_weight_update_stats(),
    }
}

pub(super) fn empty_mini_transformer_attention_weight_update_stats()
-> MiniTransformerAttentionWeightUpdateStats {
    MiniTransformerAttentionWeightUpdateStats {
        q: empty_linear_weight_update_stats(),
        k: empty_linear_weight_update_stats(),
        v: empty_linear_weight_update_stats(),
        o: empty_linear_weight_update_stats(),
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
        grad_embedding_output: Vec::new(),
    }
}

pub(super) fn add_linear_weight_update_stats_checked(
    total: &mut LinearWeightUpdateStats,
    next: LinearWeightUpdateStats,
) -> Result<(), TrainError> {
    total.gradient_saturation_count = total
        .gradient_saturation_count
        .checked_add(next.gradient_saturation_count)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_saturation"))?;
    total.zero_delta_count = total
        .zero_delta_count
        .checked_add(next.zero_delta_count)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_zero_delta"))?;
    total.weight_delta_l1 = total
        .weight_delta_l1
        .checked_add(next.weight_delta_l1)
        .ok_or(TrainError::CoreRejected("linear_weight_stats_delta_l1"))?;
    Ok(())
}

pub(super) fn add_gated_mlp_weight_update_stats_checked(
    total: &mut GatedMlpWeightUpdateStats,
    next: GatedMlpWeightUpdateStats,
) -> Result<(), TrainError> {
    add_linear_weight_update_stats_checked(&mut total.down, next.down)?;
    add_linear_weight_update_stats_checked(&mut total.up, next.up)?;
    add_linear_weight_update_stats_checked(&mut total.gate, next.gate)?;
    Ok(())
}

pub(super) fn add_mini_transformer_attention_weight_update_stats_checked(
    total: &mut MiniTransformerAttentionWeightUpdateStats,
    next: MiniTransformerAttentionWeightUpdateStats,
) -> Result<(), TrainError> {
    add_linear_weight_update_stats_checked(&mut total.q, next.q)?;
    add_linear_weight_update_stats_checked(&mut total.k, next.k)?;
    add_linear_weight_update_stats_checked(&mut total.v, next.v)?;
    add_linear_weight_update_stats_checked(&mut total.o, next.o)?;
    total.gradient_saturation_count = total
        .gradient_saturation_count
        .checked_add(next.gradient_saturation_count)
        .ok_or(TrainError::CoreRejected(
            "attention_weight_stats_saturation",
        ))?;
    total.zero_delta_count = total
        .zero_delta_count
        .checked_add(next.zero_delta_count)
        .ok_or(TrainError::CoreRejected(
            "attention_weight_stats_zero_delta",
        ))?;
    total.weight_delta_l1 = total
        .weight_delta_l1
        .checked_add(next.weight_delta_l1)
        .ok_or(TrainError::CoreRejected("attention_weight_stats_delta_l1"))?;
    Ok(())
}
