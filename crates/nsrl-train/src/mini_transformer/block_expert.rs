//! Low-rank block-expert evaluation and training.

use super::*;

pub fn mini_transformer_output_from_hidden_q15(
    model: &MiniTransformerMlpModel,
    hidden_q15: &[i16; MINI_TRANSFORMER_D_MODEL],
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let row = mini_transformer_output_row_for(&model.output_weights, hidden_q15)?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
    })
}

pub fn mini_transformer_output_gradient_to_hidden_q15(
    model: &MiniTransformerMlpModel,
    grad_output_q15: &[i16; BYTE_VOCAB],
) -> Result<[i16; MINI_TRANSFORMER_D_MODEL], TrainError> {
    let mut scaled_grad_output = [0_i32; BYTE_VOCAB];
    let mut grad_hidden_q15 = [0_i16; MINI_TRANSFORMER_D_MODEL];
    linear_backward_input_i16_i8_i16_per_channel_checked(
        grad_output_q15,
        LinearBackwardInputI16I8Params {
            weights: &model.output_weights,
            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
            grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: BYTE_VOCAB,
        },
        LinearBackwardInputWorkspace {
            scaled_grad_output: &mut scaled_grad_output,
        },
        &mut grad_hidden_q15,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_output_gradient_to_hidden",
    ))?;
    Ok(grad_hidden_q15)
}

#[derive(Debug, Clone)]
struct MiniTransformerBlockExpertLayerCache {
    base: MiniTransformerBlockForwardCache,
    latent_q15: Vec<i16>,
    adapted_output: Vec<i16>,
}

#[derive(Debug, Clone)]
struct MiniTransformerBlockExpertForwardCache {
    layers: Vec<MiniTransformerBlockExpertLayerCache>,
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
    hidden_saturation_count: usize,
}

fn block_expert_projection_sign(seed: u64, layer: usize, rank: usize, dim: usize) -> i64 {
    let mut value = seed
        ^ (layer as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (rank as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (dim as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    if value & 1 == 0 { 1 } else { -1 }
}

fn block_expert_layer_weight_range(
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
) -> Result<Range<usize>, TrainError> {
    if layer >= expert.transformer_layers {
        return Err(TrainError::InvalidConfig);
    }
    let per_layer = MINI_TRANSFORMER_D_MODEL
        .checked_mul(expert.rank)
        .ok_or(TrainError::InvalidConfig)?;
    let start = layer
        .checked_mul(per_layer)
        .ok_or(TrainError::InvalidConfig)?;
    Ok(start..start + per_layer)
}

fn block_expert_adapt_rows(
    base: &[i16],
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
) -> Result<(Vec<i16>, Vec<i16>, usize), TrainError> {
    if base.is_empty() || !base.len().is_multiple_of(MINI_TRANSFORMER_D_MODEL) {
        return Err(TrainError::InvalidConfig);
    }
    let rows = base.len() / MINI_TRANSFORMER_D_MODEL;
    let mut latent = vec![0_i16; rows * expert.rank];
    let mut output = vec![0_i16; base.len()];
    let weights = &expert.expansion_weights_q15[block_expert_layer_weight_range(expert, layer)?];
    let projection_shift = MINI_TRANSFORMER_D_MODEL.trailing_zeros() as u8;
    if 1_usize << u32::from(projection_shift) != MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidModel(
            "block expert d_model must be power of two",
        ));
    }
    let mut saturation_count = 0_usize;
    for row in 0..rows {
        let row_start = row * MINI_TRANSFORMER_D_MODEL;
        let latent_start = row * expert.rank;
        for rank in 0..expert.rank {
            let sum = (0..MINI_TRANSFORMER_D_MODEL)
                .map(|dim| {
                    i64::from(base[row_start + dim])
                        * block_expert_projection_sign(expert.projection_seed, layer, rank, dim)
                })
                .sum::<i64>();
            latent[latent_start + rank] = saturate_i16(round_shift_rhu_i64(sum, projection_shift));
        }
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let residual_acc = (0..expert.rank)
                .map(|rank| {
                    i64::from(latent[latent_start + rank])
                        * i64::from(weights[dim * expert.rank + rank])
                })
                .sum::<i64>();
            let raw = i64::from(base[row_start + dim]).saturating_add(round_shift_rhu_i64(
                residual_acc,
                Q15_SHIFT.saturating_add(expert.residual_shift),
            ));
            let adapted = saturate_i16(raw);
            saturation_count =
                saturation_count.saturating_add(usize::from(i64::from(adapted) != raw));
            output[row_start + dim] = adapted;
        }
    }
    Ok((output, latent, saturation_count))
}

fn mini_transformer_forward_with_block_expert(
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerBlockExpertForwardCache, TrainError> {
    expert.validate_for_model(model)?;
    if context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut layer_input = mini_transformer_embedding_sequence_with_position_policy_q15(
        &model.embeddings,
        &model.position_embeddings,
        context,
        position_policy,
    )?;
    let attention_weight_count = mini_transformer_attention_weight_count()?;
    let mlp_up_count = mini_transformer_mlp_up_or_gate_weight_count()?;
    let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
    let mut layer_caches = Vec::with_capacity(expert.transformer_layers);
    let mut hidden_saturation_count = 0_usize;
    for layer in 0..expert.transformer_layers {
        let attention_range = mini_transformer_layer_range(layer, attention_weight_count)?;
        let up_range = mini_transformer_layer_range(layer, mlp_up_count)?;
        let down_range = mini_transformer_layer_range(layer, mlp_down_count)?;
        let rms_range = if model.rms_norm_enabled() {
            Some(model.rms_weight_range(layer)?)
        } else {
            None
        };
        let base = mini_transformer_forward_block_for_attention_kind(
            &layer_input,
            rms_range
                .as_ref()
                .map(|range| &model.attention_rms_weights[range.clone()]),
            rms_range
                .as_ref()
                .map(|range| &model.mlp_rms_weights[range.clone()]),
            &model.q_weights[attention_range.clone()],
            &model.k_weights[attention_range.clone()],
            &model.v_weights[attention_range.clone()],
            &model.o_weights[attention_range],
            &model.up_weights[up_range.clone()],
            &model.gate_weights[up_range],
            &model.down_weights[down_range],
            attention_kind,
        )?;
        let (adapted_output, latent_q15, saturations) =
            block_expert_adapt_rows(&base.block_output, expert, layer)?;
        hidden_saturation_count = hidden_saturation_count.saturating_add(saturations);
        layer_input = adapted_output.clone();
        layer_caches.push(MiniTransformerBlockExpertLayerCache {
            base,
            latent_q15,
            adapted_output,
        });
    }
    let last = layer_input
        .len()
        .checked_sub(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let mut output_features = [0_i16; MINI_TRANSFORMER_D_MODEL];
    output_features.copy_from_slice(&layer_input[last..last + MINI_TRANSFORMER_D_MODEL]);
    let row = mini_transformer_output_row_for(&model.output_weights, &output_features)?;
    #[cfg(feature = "mini-calibrated")]
    let mut row = row;
    #[cfg(feature = "mini-calibrated")]
    if position_policy == MiniTransformerPositionPolicy::Nope {
        if let Some(predicted) =
            mini_transformer_ngram_cache_prediction(&model.position_embeddings, context)
        {
            mini_transformer_rerank_output_row(&mut row, predicted)?;
        }
    }
    Ok(MiniTransformerBlockExpertForwardCache {
        layers: layer_caches,
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
        hidden_saturation_count,
    })
}

fn block_expert_backward_rows(
    expert: &MiniTransformerBlockLowRankExpert,
    layer: usize,
    cache: &MiniTransformerBlockExpertLayerCache,
    grad_adapted: &[i16],
    gradient_accumulators: &mut [i64],
) -> Result<Vec<i16>, TrainError> {
    if grad_adapted.len() != cache.base.block_output.len()
        || cache.adapted_output.len() != grad_adapted.len()
        || cache.latent_q15.len() * MINI_TRANSFORMER_D_MODEL != grad_adapted.len() * expert.rank
    {
        return Err(TrainError::InvalidConfig);
    }
    let range = block_expert_layer_weight_range(expert, layer)?;
    if gradient_accumulators.len() != expert.expansion_weights_q15.len() {
        return Err(TrainError::InvalidConfig);
    }
    let weights = &expert.expansion_weights_q15[range.clone()];
    let gradients = &mut gradient_accumulators[range];
    let rows = grad_adapted.len() / MINI_TRANSFORMER_D_MODEL;
    let projection_shift = MINI_TRANSFORMER_D_MODEL.trailing_zeros() as u8;
    let mut grad_base = vec![0_i16; grad_adapted.len()];
    for row in 0..rows {
        let row_start = row * MINI_TRANSFORMER_D_MODEL;
        let latent_start = row * expert.rank;
        let mut grad_latent = vec![0_i64; expert.rank];
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let index = row_start + dim;
            let grad = if cache.adapted_output[index] == i16::MIN
                || cache.adapted_output[index] == i16::MAX
            {
                0_i64
            } else {
                i64::from(grad_adapted[index])
            };
            for (rank, grad_latent_value) in grad_latent.iter_mut().enumerate() {
                let weight_index = dim * expert.rank + rank;
                gradients[weight_index] = gradients[weight_index].saturating_add(
                    grad.saturating_mul(i64::from(cache.latent_q15[latent_start + rank])),
                );
                *grad_latent_value = grad_latent_value.saturating_add(round_shift_rhu_i64(
                    grad.saturating_mul(i64::from(weights[weight_index])),
                    Q15_SHIFT.saturating_add(expert.residual_shift),
                ));
            }
        }
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let projected = (0..expert.rank)
                .map(|rank| {
                    grad_latent[rank]
                        * block_expert_projection_sign(expert.projection_seed, layer, rank, dim)
                })
                .sum::<i64>();
            grad_base[row_start + dim] = saturate_i16(
                i64::from(grad_adapted[row_start + dim])
                    .saturating_add(round_shift_rhu_i64(projected, projection_shift)),
            );
        }
    }
    Ok(grad_base)
}

pub fn mini_transformer_next_token_row_with_block_expert(
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let cache = mini_transformer_forward_with_block_expert(
        model,
        expert,
        context,
        attention_kind,
        position_policy,
    )?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: cache.logits_q8,
        probabilities_q15: cache.probabilities_q15,
    })
}

pub fn evaluate_mini_transformer_block_expert(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpEvalConfig,
) -> Result<MiniTransformerBlockExpertMetrics, TrainError> {
    expert.validate_for_model(model)?;
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || config.attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: config.seq_len,
            stride: config.stride,
            max_windows: config.max_windows,
            attention_kind: config.attention_kind,
            position_policy: config.position_policy,
            ..MiniTransformerMlpTrainConfig::default()
        },
    );
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut metrics = MiniTransformerBlockExpertMetrics {
        windows: starts.len(),
        mistakes: 0,
        probability_error_q15: 0,
        hidden_saturation_count: 0,
    };
    for start in starts {
        let end = start + config.seq_len;
        let cache = mini_transformer_forward_with_block_expert(
            model,
            expert,
            &tokens[start..end],
            config.attention_kind,
            config.position_policy,
        )?;
        metrics.mistakes = metrics.mistakes.saturating_add(usize::from(
            byte_argmax_i32(&cache.logits_q8) != tokens[end],
        ));
        metrics.probability_error_q15 =
            metrics
                .probability_error_q15
                .saturating_add(byte_sample_probability_error_q15(
                    &cache.probabilities_q15,
                    tokens[end],
                ));
        metrics.hidden_saturation_count = metrics
            .hidden_saturation_count
            .saturating_add(cache.hidden_saturation_count);
    }
    Ok(metrics)
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    train_mini_transformer_block_expert_with_layer_scope(
        tokens,
        model,
        expert,
        config,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert_with_layer_scope(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
    train_layer: Option<usize>,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
        tokens,
        model,
        expert,
        config,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        train_layer,
        false,
        MiniTransformerBlockExpertObjective::CrossEntropy,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
    train_layer: Option<usize>,
    bidirectional_loss_guard: bool,
    objective: MiniTransformerBlockExpertObjective,
) -> Result<MiniTransformerBlockExpertTrainStats, TrainError> {
    expert.validate_for_model(model)?;
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.seq_len != model.context_seq_len
        || batch_windows == 0
        || learning_rate <= 0
        || learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind.uses_incremental_state()
        || train_layer.is_some_and(|layer| layer >= expert.transformer_layers)
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut update_residuals = vec![0_i64; expert.expansion_weights_q15.len()];
    let mut stats = MiniTransformerBlockExpertTrainStats {
        optimizer_steps: 0,
        accepted_forward_steps: 0,
        accepted_reverse_steps: 0,
        rejected_steps: 0,
        weight_delta_l1: 0,
        weight_saturation_count: 0,
        hidden_saturation_count: 0,
    };
    let last_start = (config.seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let mut workspace = MiniTransformerHostTrainCoreWorkspaceBuffers::new(config.seq_len)?;
    let mut frozen_model = model.clone();
    for _ in 0..config.epochs {
        for batch in starts.chunks(batch_windows) {
            let mut gradients = vec![0_i64; expert.expansion_weights_q15.len()];
            for &start in batch {
                let end = start + config.seq_len;
                let cache = mini_transformer_forward_with_block_expert(
                    model,
                    expert,
                    &tokens[start..end],
                    config.attention_kind,
                    config.position_policy,
                )?;
                stats.hidden_saturation_count = stats
                    .hidden_saturation_count
                    .saturating_add(cache.hidden_saturation_count);
                let mut grad_output =
                    byte_vocab_softmax_gradient_q15(&cache.probabilities_q15, tokens[end]);
                if objective == MiniTransformerBlockExpertObjective::ProbabilityError {
                    let target_probability =
                        i64::from(cache.probabilities_q15[usize::from(tokens[end])].max(0));
                    for gradient in &mut grad_output {
                        *gradient = (i64::from(*gradient).saturating_mul(target_probability)
                            / i64::from(i16::MAX))
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                            as i32;
                    }
                }
                apply_byte_argmax_margin_gradient_q15(
                    &mut grad_output,
                    &cache.logits_q8,
                    tokens[end],
                    config.argmax_margin_weight_q15,
                );
                let grad_output_q15 = byte_gradient_i32_to_i16(&grad_output);
                let grad_last =
                    mini_transformer_output_gradient_to_hidden_q15(model, &grad_output_q15)?;
                let mut grad_adapted = vec![0_i16; config.seq_len * MINI_TRANSFORMER_D_MODEL];
                grad_adapted[last_start..last_start + MINI_TRANSFORMER_D_MODEL]
                    .copy_from_slice(&grad_last);
                let mut dummy =
                    MiniTransformerMapReduceBatchResult::new(config, expert.transformer_layers)?;
                for layer in (0..expert.transformer_layers).rev() {
                    let grad_base = block_expert_backward_rows(
                        expert,
                        layer,
                        &cache.layers[layer],
                        &grad_adapted,
                        &mut gradients,
                    )?;
                    let layer_config = mini_transformer_stacked_layer_runtime_config(
                        config,
                        layer,
                        expert.transformer_layers,
                    );
                    let backward = mini_transformer_block_backward_accumulate_i64_checked(
                        &cache.layers[layer].base,
                        &grad_base,
                        &mut frozen_model,
                        layer,
                        layer_config,
                        &mut workspace,
                        &mut dummy.mlp_weight_gradients[layer],
                        &mut dummy.attention_weight_gradients[layer],
                        &mut dummy.rms_weight_gradients[layer],
                    )?;
                    grad_adapted = backward.grad_input;
                }
            }
            let gradient_shift = learning_rate_shift
                .checked_add(Q15_SHIFT)
                .and_then(|shift| shift.checked_add(expert.residual_shift))
                .ok_or(TrainError::InvalidConfig)?;
            let denominator = i64::try_from(batch.len())
                .map_err(|_| TrainError::InvalidConfig)?
                .checked_shl(u32::from(gradient_shift))
                .ok_or(TrainError::InvalidConfig)?;
            let parameters_per_layer = MINI_TRANSFORMER_D_MODEL
                .checked_mul(expert.rank)
                .ok_or(TrainError::InvalidConfig)?;
            if bidirectional_loss_guard {
                let baseline_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                let original_weights = expert.expansion_weights_q15.clone();
                let original_residuals = update_residuals.clone();
                let mut forward_weights = original_weights.clone();
                let mut reverse_weights = original_weights.clone();
                let mut next_residuals = original_residuals.clone();
                let mut forward_saturations = vec![false; original_weights.len()];
                let mut reverse_saturations = vec![false; original_weights.len()];
                for index in 0..original_weights.len() {
                    if train_layer.is_some_and(|layer| index / parameters_per_layer != layer) {
                        continue;
                    }
                    let numerator = gradients[index]
                        .saturating_mul(learning_rate)
                        .saturating_add(original_residuals[index]);
                    let averaged = round_div_signed_i64(numerator, denominator)?;
                    next_residuals[index] =
                        numerator.saturating_sub(averaged.saturating_mul(denominator));
                    let update = averaged;
                    let previous = i64::from(original_weights[index]);
                    let forward_raw = previous.saturating_sub(update);
                    let reverse_raw = previous.saturating_add(update);
                    forward_weights[index] = saturate_i16(forward_raw);
                    reverse_weights[index] = saturate_i16(reverse_raw);
                    forward_saturations[index] = i64::from(forward_weights[index]) != forward_raw;
                    reverse_saturations[index] = i64::from(reverse_weights[index]) != reverse_raw;
                }
                expert.expansion_weights_q15 = forward_weights.clone();
                let forward_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                expert.expansion_weights_q15 = reverse_weights.clone();
                let reverse_error = mini_transformer_block_expert_batch_error(
                    tokens, batch, model, expert, config,
                )?;
                let (selected, selected_saturations) = if forward_error < baseline_error
                    && forward_error <= reverse_error
                {
                    stats.accepted_forward_steps = stats.accepted_forward_steps.saturating_add(1);
                    (Some(forward_weights), Some(forward_saturations))
                } else if reverse_error < baseline_error && reverse_error < forward_error {
                    stats.accepted_reverse_steps = stats.accepted_reverse_steps.saturating_add(1);
                    (Some(reverse_weights), Some(reverse_saturations))
                } else {
                    stats.rejected_steps = stats.rejected_steps.saturating_add(1);
                    (None, None)
                };
                if let Some(selected) = selected {
                    for (index, (&previous, &next)) in
                        original_weights.iter().zip(selected.iter()).enumerate()
                    {
                        stats.weight_delta_l1 = stats
                            .weight_delta_l1
                            .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                        if selected_saturations
                            .as_ref()
                            .is_some_and(|values| values[index])
                        {
                            stats.weight_saturation_count =
                                stats.weight_saturation_count.saturating_add(1);
                            next_residuals[index] = 0;
                        }
                    }
                    expert.expansion_weights_q15 = selected;
                    update_residuals = next_residuals;
                } else {
                    expert.expansion_weights_q15 = original_weights;
                    update_residuals = original_residuals;
                }
            } else {
                for index in 0..expert.expansion_weights_q15.len() {
                    if train_layer.is_some_and(|layer| index / parameters_per_layer != layer) {
                        continue;
                    }
                    let numerator = gradients[index]
                        .saturating_mul(learning_rate)
                        .saturating_add(update_residuals[index]);
                    let averaged = round_div_signed_i64(numerator, denominator)?;
                    update_residuals[index] =
                        numerator.saturating_sub(averaged.saturating_mul(denominator));
                    let update = averaged;
                    let previous = expert.expansion_weights_q15[index];
                    let raw = i64::from(previous).saturating_sub(update);
                    let next = saturate_i16(raw);
                    if i64::from(next) != raw {
                        stats.weight_saturation_count =
                            stats.weight_saturation_count.saturating_add(1);
                        update_residuals[index] = 0;
                    }
                    stats.weight_delta_l1 = stats
                        .weight_delta_l1
                        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                    expert.expansion_weights_q15[index] = next;
                }
                stats.accepted_forward_steps = stats.accepted_forward_steps.saturating_add(1);
            }
            stats.optimizer_steps = stats.optimizer_steps.saturating_add(1);
        }
    }
    Ok(stats)
}

fn mini_transformer_block_expert_batch_error(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    expert: &MiniTransformerBlockLowRankExpert,
    config: MiniTransformerMlpTrainConfig,
) -> Result<usize, TrainError> {
    let mut error = 0_usize;
    for &start in starts {
        let end = start
            .checked_add(config.seq_len)
            .ok_or(TrainError::InvalidConfig)?;
        if end >= tokens.len() {
            return Err(TrainError::InvalidConfig);
        }
        let cache = mini_transformer_forward_with_block_expert(
            model,
            expert,
            &tokens[start..end],
            config.attention_kind,
            config.position_policy,
        )?;
        error = error.saturating_add(byte_sample_probability_error_q15(
            &cache.probabilities_q15,
            tokens[end],
        ));
    }
    Ok(error)
}

fn round_div_signed_i64(value: i64, denominator: i64) -> Result<i64, TrainError> {
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    Ok(if value >= 0 {
        value.saturating_add(half) / denominator
    } else {
        value.saturating_sub(half) / denominator
    })
}
