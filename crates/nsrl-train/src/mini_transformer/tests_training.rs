//! Mini-transformer tests — train.
use super::*;

#[test]
fn mini_transformer_embedding_sequence_includes_trainable_position_embedding() {
    let mut embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    let position_embeddings = initial_mini_transformer_position_embeddings(2);
    let row_start = usize::from(b'a') * MINI_TRANSFORMER_D_MODEL;
    embeddings[row_start..row_start + MINI_TRANSFORMER_D_MODEL].fill(256);

    let sequence = mini_transformer_embedding_sequence_with_position_policy_q15(
        &embeddings,
        &position_embeddings,
        b"aa",
        MiniTransformerPositionPolicy::LearnedAbsolute,
    )
    .expect("sequence");
    let first = &sequence[..MINI_TRANSFORMER_D_MODEL];
    let second = &sequence[MINI_TRANSFORMER_D_MODEL..2 * MINI_TRANSFORMER_D_MODEL];

    assert_ne!(first, second);
    assert!(sequence.iter().all(|&value| (-768..=1280).contains(&value)));
}

#[test]
fn mini_transformer_nope_embedding_sequence_skips_position_embedding() {
    let mut embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    let position_embeddings = initial_mini_transformer_position_embeddings(2);
    let row_start = usize::from(b'a') * MINI_TRANSFORMER_D_MODEL;
    embeddings[row_start..row_start + MINI_TRANSFORMER_D_MODEL].fill(256);

    let sequence = mini_transformer_embedding_sequence_with_position_policy_q15(
        &embeddings,
        &position_embeddings,
        b"aa",
        MiniTransformerPositionPolicy::Nope,
    )
    .expect("sequence");
    let first = &sequence[..MINI_TRANSFORMER_D_MODEL];
    let second = &sequence[MINI_TRANSFORMER_D_MODEL..2 * MINI_TRANSFORMER_D_MODEL];

    assert_eq!(first, second);
    assert!(sequence.iter().all(|&value| value == 256));
}

#[cfg(not(feature = "mini-calibrated"))]
#[test]
fn mini_transformer_mlp_training_updates_head_mlp_and_attention() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 2,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(64),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
    )
    .expect("mini train");

    assert_eq!(trace.token_count, tokens.len());
    assert!(trace.windows > 0);
    assert!(trace.updates > 0);
    assert!(trace.initial_probability_error_q15 > trace.final_probability_error_q15);
    assert_ne!(trace.initial_model_hash, trace.final_model_hash);
    assert_ne!(trace.initial_embedding_hash, trace.final_embedding_hash);
    assert_ne!(trace.initial_output_head_hash, trace.final_output_head_hash);
    assert_ne!(trace.initial_mlp_hash, trace.final_mlp_hash);
    assert_ne!(trace.initial_attention_hash, trace.final_attention_hash);
    assert_eq!(trace.output_head_saturation_count, 0);
    assert!(trace.output_head_delta_l1 > 0);
    assert!(trace.mlp_delta_l1 > 0);
    assert!(trace.embedding_delta_l1 > 0);
    assert!(trace.attention_delta_l1 > 0);
    assert!(
        trace
            .steps
            .iter()
            .any(|step| step.mlp_hash_before != step.mlp_hash_after)
    );
    assert!(
        trace
            .steps
            .iter()
            .any(|step| step.attention_hash_before != step.attention_hash_after)
    );
}

#[test]
fn mini_transformer_stacked_serial_training_updates_lower_layer() {
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    assert_eq!(model.transformer_layers(), 2);
    let context = b"To b";
    let cache = mini_transformer_forward_for_attention_and_position(
        &model,
        context,
        MiniTransformerAttentionKind::Base2Softmax,
        MiniTransformerPositionPolicy::LearnedAbsolute,
    )
    .expect("stacked forward");
    let first_attention_range = model
        .attention_weight_range(0)
        .expect("first attention range");
    let first_down_range = model.mlp_down_weight_range(0).expect("first down range");
    let initial_first_o_hash = hash_i8_slice(&model.o_weights[first_attention_range.clone()]);
    let initial_first_down_hash = hash_i8_slice(&model.down_weights[first_down_range.clone()]);
    let mut workspace =
        MiniTransformerHostTrainCoreWorkspaceBuffers::new(context.len()).expect("workspace");
    let mut grad_block_output = vec![0_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
    for (index, gradient) in grad_block_output.iter_mut().enumerate() {
        *gradient = if index.is_multiple_of(3) { 2048 } else { -1024 };
    }

    let update = mini_transformer_block_backward_update_i8_checked(
        &cache.layers[0],
        &grad_block_output,
        &mut model,
        0,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: context.len(),
            stride: 1,
            window_offset: 0,
            max_windows: Some(1),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: 2,
            output_learning_rate_shift: 16,
            mlp_learning_rate_shift: 10,
            embedding_learning_rate_shift: 12,
            attention_learning_rate_shift: 10,
            attention_q_learning_rate_shift: 10,
            attention_qk_learning_rate_shift: 10,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
        &mut workspace,
    )
    .expect("lower block backward");

    assert!(update.mlp_update.weight_delta_l1().unwrap_or(0) > 0);
    assert!(update.attention_update.weight_delta_l1 > 0);
    assert_ne!(
        hash_i8_slice(&model.down_weights[first_down_range]),
        initial_first_down_hash
    );
    assert_ne!(
        hash_i8_slice(&model.o_weights[first_attention_range]),
        initial_first_o_hash
    );
    assert_eq!(
        update.grad_input.len(),
        context.len() * MINI_TRANSFORMER_D_MODEL
    );
}

#[test]
fn linear_attention_backward_produces_qkv_gradients() {
    let seq_len = 2;
    let total = seq_len * MINI_TRANSFORMER_D_MODEL;
    let mut q = vec![0_i16; total];
    let mut k = vec![0_i16; total];
    let mut v = vec![0_i16; total];
    let mut grad_context = vec![0_i16; total];
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        q[dim] = 256 + dim as i16 * 8;
        q[MINI_TRANSFORMER_D_MODEL + dim] = -128 + dim as i16 * 4;
        k[dim] = -192 + dim as i16 * 6;
        k[MINI_TRANSFORMER_D_MODEL + dim] = 224 - dim as i16 * 5;
        v[dim] = 8192 - dim as i16 * 16;
        v[MINI_TRANSFORMER_D_MODEL + dim] = -6144 + dim as i16 * 12;
        grad_context[dim] = 4096 + dim as i16 * 8;
        grad_context[MINI_TRANSFORMER_D_MODEL + dim] = -3072 + dim as i16 * 6;
    }

    let (grad_q, grad_k, grad_v) =
        mini_transformer_linear_attention_qkv_gradients_q15(seq_len, &q, &k, &v, &grad_context)
            .expect("linear gradients");

    assert_eq!(grad_q.len(), total);
    assert_eq!(grad_k.len(), total);
    assert_eq!(grad_v.len(), total);
    assert!(grad_q.iter().any(|&value| value != 0));
    assert!(grad_k.iter().any(|&value| value != 0));
    assert!(grad_v.iter().any(|&value| value != 0));
}

#[cfg(not(feature = "mini-calibrated"))]
#[test]
fn mini_transformer_mlp_training_can_use_linear_attention() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(16),
            batch_windows: 1,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 13,
            attention_qk_learning_rate_shift: 16,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
    )
    .expect("linear mini train");

    assert!(trace.updates > 0);
    assert_eq!(trace.final_invalid_forward_count, 0);
    assert!(trace.attention_delta_l1 > 0);
    assert_ne!(trace.initial_attention_hash, trace.final_attention_hash);
    let line = trace.to_json_line();
    assert!(line.contains("\"attention_kind\":\"linear\""));
    assert!(line.contains(
        "\"attention_backward\":\"linear_numerator_straight_through_denominator_constant\""
    ));
    assert!(line.contains("\"attention_q_learning_rate_shift\":13"));
}

#[test]
fn mini_transformer_adaptive_attention_shifts_are_traced() {
    let tokens = b"to be or not to be to be or not to be ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 2,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 22,
            attention_qk_learning_rate_shift: 22,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: true,
            adaptive_holographic_shifts: true,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
    )
    .expect("adaptive train");

    assert!(trace.adaptive_holographic_update_count > 0);
    assert!(
        trace.adaptive_holographic_update_count
            >= trace.adaptive_attention_holographic_update_count
    );
    assert!(trace.adaptive_attention_holographic_update_count > 0);
    assert!(trace.final_output_learning_rate_shift <= MAX_RIGHT_SHIFT);
    assert!(trace.final_mlp_learning_rate_shift <= MAX_RIGHT_SHIFT);
    assert!(trace.final_embedding_learning_rate_shift <= MAX_RIGHT_SHIFT);
    assert!(trace.final_attention_q_learning_rate_shift <= MAX_RIGHT_SHIFT);
    assert!(trace.final_attention_qk_learning_rate_shift <= MAX_RIGHT_SHIFT);
    assert!(trace.final_attention_learning_rate_shift <= MAX_RIGHT_SHIFT);
    let line = trace.to_json_line();
    assert!(line.contains("\"adaptive_attention_shifts\":true"));
    assert!(line.contains("\"adaptive_holographic_shifts\":true"));
    assert!(line.contains("\"adaptive_holographic_update_count\":"));
    assert!(line.contains("\"adaptive_holographic_meta_dim\":8"));
    assert!(line.contains("\"adaptive_holographic_action_count\":5"));
    assert!(line.contains("\"adaptive_attention_holographic_update_count\":"));
    assert!(line.contains("\"final_output_learning_rate_shift\":"));
    assert!(line.contains("\"final_attention_q_learning_rate_shift\":"));
}

#[test]
fn mini_transformer_adaptive_holographic_shifts_enable_controller() {
    let tokens = b"to be or not to be to be or not to be ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 2,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 22,
            attention_qk_learning_rate_shift: 22,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: true,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
    )
    .expect("holographic adaptive train");

    assert!(trace.adaptive_holographic_update_count > 0);
    assert!(trace.adaptive_attention_holographic_update_count > 0);
    let line = trace.to_json_line();
    assert!(line.contains("\"adaptive_attention_shifts\":false"));
    assert!(line.contains("\"adaptive_holographic_shifts\":true"));
    assert!(line.contains("\"adaptive_holographic_meta_dim\":8"));
    assert!(line.contains("\"adaptive_holographic_hash\":"));
}

#[test]
fn mini_transformer_holographic_memory_binds_previous_state_to_next_teacher() {
    let mut memory = IntegerHolographicShiftMemory::new();
    let mut previous_state = None;
    let state_a = [i16::MAX, 1024, 0, 512, 0, 0, 256, -512];
    let state_b = [i16::MAX, 2048, 256, 768, 0, 0, 512, 0];

    mini_transformer_holo_remember_lagged(&mut memory, &mut previous_state, state_a, -1);
    assert_eq!(memory.update_count, 0);
    assert_eq!(previous_state, Some(state_a));

    mini_transformer_holo_remember_lagged(&mut memory, &mut previous_state, state_b, 1);
    assert_eq!(memory.update_count, 1);
    assert_eq!(previous_state, Some(state_b));
    assert_eq!(memory.retrieve_delta(&state_a), Some(1));
}

#[test]
fn mini_transformer_holographic_memory_can_act_when_teacher_is_silent() {
    assert_eq!(mini_transformer_holo_safety_delta(0, -2, false), -1);
    assert_eq!(mini_transformer_holo_safety_delta(0, -1, false), -1);
    assert_eq!(mini_transformer_holo_safety_delta(0, 0, false), 0);
    assert_eq!(mini_transformer_holo_safety_delta(0, 1, false), 1);
    assert_eq!(mini_transformer_holo_safety_delta(0, 2, false), 1);
    assert_eq!(mini_transformer_holo_safety_delta(1, -1, true), 1);
    assert_eq!(mini_transformer_holo_safety_delta(-1, 1, true), -1);
    assert_eq!(mini_transformer_holo_safety_delta(1, -1, false), 0);
    assert_eq!(mini_transformer_holo_safety_delta(-1, 1, false), 0);
}

#[test]
fn mini_transformer_holographic_authority_requires_history_and_cooldown() {
    let mut last_adjust_batch = None;
    assert_eq!(
        mini_transformer_holo_authorized_delta(
            -1,
            0,
            MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES - 1,
            1,
            &mut last_adjust_batch,
        ),
        0
    );
    assert_eq!(last_adjust_batch, None);
    assert_eq!(
        mini_transformer_holo_authorized_delta(
            -1,
            0,
            MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES,
            8,
            &mut last_adjust_batch,
        ),
        -1
    );
    assert_eq!(last_adjust_batch, Some(8));
    assert_eq!(
        mini_transformer_holo_authorized_delta(
            1,
            1,
            0,
            8 + MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES - 1,
            &mut last_adjust_batch,
        ),
        0
    );
    assert_eq!(
        mini_transformer_holo_authorized_delta(
            1,
            1,
            0,
            8 + MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES,
            &mut last_adjust_batch,
        ),
        1
    );
}

#[test]
fn mini_transformer_adaptive_rule_shifts_emit_events() {
    let tokens = b"to be or not to be to be or not to be ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 2,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 22,
            attention_qk_learning_rate_shift: 22,
            adaptive_rule_shifts: true,
            adaptive_rule_interval_batches: 1,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        },
    )
    .expect("rule adaptive train");

    assert!(trace.adaptive_rule_update_count > 0);
    assert!(trace.adaptive_rule_shift_adjustment_count > 0);
    assert_eq!(trace.adaptive_holographic_update_count, 0);
    assert_eq!(trace.adaptive_holographic_shift_adjustment_count, 0);
    assert!(!trace.adaptive_shift_events.is_empty());
    let line = trace.to_json_line();
    assert!(line.contains("\"adaptive_rule_shifts\":true"));
    assert!(line.contains("\"adaptive_rule_interval_batches\":1"));
    assert!(line.contains("\"adaptive_rule_shift_adjustment_count\":"));
    assert!(line.contains("\"adaptive_shift_events\":["));
    assert!(line.contains("\"component\":\""));
    assert!(line.contains("\"reason\":\""));
}

#[test]
fn mini_transformer_rule_saturation_is_window_gated() {
    let interval = 4;
    let weight_count = 512;
    let quiet_stats = LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 1,
    };
    let sparse_saturation = LinearWeightUpdateStats {
        gradient_saturation_count: 1,
        zero_delta_count: 0,
        weight_delta_l1: 1,
    };

    let mut sparse_window = MiniTransformerRuleShiftWindow::new();
    sparse_window.observe_accepted(sparse_saturation);
    assert_eq!(
        mini_transformer_rule_generic_delta(sparse_window, weight_count, interval),
        None
    );
    assert!(!mini_transformer_rule_should_reset(sparse_window, interval));
    for _ in 1..interval {
        sparse_window.observe_accepted(quiet_stats);
    }
    assert_eq!(
        mini_transformer_rule_generic_delta(sparse_window, weight_count, interval),
        None
    );
    assert!(mini_transformer_rule_should_reset(sparse_window, interval));

    let mut pressure_window = MiniTransformerRuleShiftWindow::new();
    for _ in 0..interval {
        pressure_window.observe_accepted(sparse_saturation);
    }
    assert_eq!(
        mini_transformer_rule_generic_delta(pressure_window, weight_count, interval),
        Some((1, "saturation"))
    );
}

#[test]
fn mini_transformer_qk_rule_prioritizes_dead_gradient_over_saturation() {
    let interval = 4;
    let mut q_window = MiniTransformerRuleShiftWindow::new();
    let mut k_window = MiniTransformerRuleShiftWindow::new();
    let dead_saturating_q = LinearWeightUpdateStats {
        gradient_saturation_count: 1024,
        zero_delta_count: mini_transformer_attention_projection_weight_count(),
        weight_delta_l1: 0,
    };
    let moving_k = LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 100_000,
    };

    for _ in 0..interval {
        q_window.observe_accepted(dead_saturating_q);
        k_window.observe_accepted(moving_k);
    }

    assert_eq!(
        mini_transformer_rule_q_delta(q_window, k_window, interval),
        Some((-1, "zero_delta"))
    );
}

#[test]
fn mini_transformer_k_rule_prioritizes_dead_gradient_over_saturation() {
    let interval = 4;
    let mut k_window = MiniTransformerRuleShiftWindow::new();
    let mut q_window = MiniTransformerRuleShiftWindow::new();
    let dead_saturating_k = LinearWeightUpdateStats {
        gradient_saturation_count: 1024,
        zero_delta_count: mini_transformer_attention_projection_weight_count(),
        weight_delta_l1: 0,
    };
    let moving_q = LinearWeightUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 100_000,
    };

    for _ in 0..interval {
        k_window.observe_accepted(dead_saturating_k);
        q_window.observe_accepted(moving_q);
    }

    assert_eq!(
        mini_transformer_rule_k_delta(k_window, q_window, interval),
        Some((-1, "zero_delta"))
    );
}
