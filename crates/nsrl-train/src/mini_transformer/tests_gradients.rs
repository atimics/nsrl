//! Mini-transformer tests — grad.
use super::*;

#[test]
fn linear_weight_gradient_i64_averages_then_updates_i8() {
    let mut gradient = LinearWeightGradientI64::new(2, 2).expect("gradient");
    let input = [4096_i16, 8192_i16];
    let scaled_grad_output = [1024_i32, 2048_i32];

    accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
        .expect("first sample");
    accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
        .expect("second sample");

    let mut weights = [10_i8, 10_i8, 10_i8, 10_i8];
    let stats = apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 22, false)
        .expect("apply");

    assert_eq!(weights, [9, 8, 8, 6]);
    assert_eq!(stats.gradient_saturation_count, 0);
    assert_eq!(stats.zero_delta_count, 0);
    assert_eq!(stats.weight_delta_l1, 9);
    assert_eq!(gradient.sample_count, 0);
    assert!(gradient.accumulators.iter().all(|&value| value == 0));
}

#[test]
fn linear_weight_gradient_i64_carries_subthreshold_residuals() {
    let mut gradient = LinearWeightGradientI64::new(1, 1).expect("gradient");
    let input = [1_i16];
    let scaled_grad_output = [1_i32];
    let mut weights = [10_i8];

    accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
        .expect("first sample");
    let first = apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 2, true)
        .expect("first apply");

    assert_eq!(weights, [10]);
    assert_eq!(first.zero_delta_count, 1);
    assert_eq!(first.weight_delta_l1, 0);
    assert_eq!(gradient.residuals, [1]);

    accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
        .expect("second sample");
    let second = apply_linear_weight_gradient_i64_to_i8(&mut gradient, &mut weights, 1, 2, true)
        .expect("second apply");

    assert_eq!(weights, [9]);
    assert_eq!(second.zero_delta_count, 0);
    assert_eq!(second.weight_delta_l1, 1);
    assert_eq!(gradient.residuals, [-2]);
}

#[test]
fn mini_transformer_effective_batch_shift_compensates_batch_average() {
    assert_eq!(
        mini_transformer_component_shift_for_effective_batch_shift(18, 8).expect("shift"),
        15
    );
    assert_eq!(
        mini_transformer_component_shift_for_effective_batch_shift(3, 8).expect("shift"),
        0
    );
    assert!(mini_transformer_component_shift_for_effective_batch_shift(2, 8).is_err());
}

#[test]
fn linear_weight_gradient_i64_uses_effective_batch_shift() {
    let mut gradient = LinearWeightGradientI64::new(1, 1).expect("gradient");
    let input = [1_i16];
    let scaled_grad_output = [1_i32];
    for _ in 0..8 {
        accumulate_linear_weight_gradient_i64_prescaled(&input, &scaled_grad_output, &mut gradient)
            .expect("sample");
    }
    let mut compensated_weights = [10_i8];
    let compensated_shift =
        mini_transformer_component_shift_for_effective_batch_shift(3, 8).expect("shift");
    let compensated = apply_linear_weight_gradient_i64_to_i8(
        &mut gradient,
        &mut compensated_weights,
        1,
        compensated_shift,
        true,
    )
    .expect("compensated apply");

    assert_eq!(compensated_weights, [9]);
    assert_eq!(compensated.zero_delta_count, 0);
    assert_eq!(compensated.weight_delta_l1, 1);

    let mut uncompensated = LinearWeightGradientI64::new(1, 1).expect("gradient");
    for _ in 0..8 {
        accumulate_linear_weight_gradient_i64_prescaled(
            &input,
            &scaled_grad_output,
            &mut uncompensated,
        )
        .expect("sample");
    }
    let mut uncompensated_weights = [10_i8];
    let uncompensated_stats = apply_linear_weight_gradient_i64_to_i8(
        &mut uncompensated,
        &mut uncompensated_weights,
        1,
        3,
        true,
    )
    .expect("uncompensated apply");

    assert_eq!(uncompensated_weights, [10]);
    assert_eq!(uncompensated_stats.zero_delta_count, 1);
    assert_eq!(uncompensated_stats.weight_delta_l1, 0);
    assert_eq!(uncompensated.residual_l1(), 1);
}

#[test]
fn gated_mlp_weight_gradient_i64_averages_then_updates_i8() {
    let scales = [FixedScale {
        multiplier: 1,
        right_shift: 0,
    }; 2];
    let input = [4096_i16, 8192_i16];
    let grad_output = [1024_i16, -1024_i16];
    let forward_gated = [4096_i16, -4096_i16];
    let grad_up = [2048_i16, -1024_i16];
    let grad_gate = [-2048_i16, 1024_i16];
    let params = GatedMlpWeightUpdateParams {
        up_scales: &scales,
        gate_scales: &scales,
        down_scales: &scales,
        down_to_hidden_scales: &scales,
        seq_len: 1,
        d_model: 2,
        hidden_dim: 2,
        learning_rate: 1,
        learning_rate_shift: 22,
    };
    let mut gradient = GatedMlpWeightGradientI64::new(2, 2).expect("gradient");
    let mut scaled = [0_i32; 2];

    accumulate_gated_mlp_weight_gradient_i64(
        &input,
        &grad_output,
        &forward_gated,
        &grad_up,
        &grad_gate,
        params,
        &mut gradient,
        &mut scaled,
    )
    .expect("first sample");
    accumulate_gated_mlp_weight_gradient_i64(
        &input,
        &grad_output,
        &forward_gated,
        &grad_up,
        &grad_gate,
        params,
        &mut gradient,
        &mut scaled,
    )
    .expect("second sample");

    let mut up_weights = [10_i8; 4];
    let mut gate_weights = [10_i8; 4];
    let mut down_weights = [10_i8; 4];
    let stats = apply_gated_mlp_weight_gradient_i64_to_i8(
        &mut gradient,
        &mut up_weights,
        &mut gate_weights,
        &mut down_weights,
        1,
        22,
        false,
    )
    .expect("apply");

    assert_eq!(down_weights, [9, 11, 11, 9]);
    assert_eq!(up_weights, [8, 6, 11, 12]);
    assert_eq!(gate_weights, [12, 14, 9, 8]);
    assert_eq!(stats.gradient_saturation_count(), Some(0));
    assert_eq!(stats.zero_delta_count(), Some(0));
    assert_eq!(stats.weight_delta_l1(), Some(22));
    assert_eq!(gradient.down.sample_count, 0);
    assert_eq!(gradient.up.sample_count, 0);
    assert_eq!(gradient.gate.sample_count, 0);
}

#[cfg(not(feature = "mini-calibrated"))]
#[test]
fn attention_weight_gradient_i64_averages_then_updates_i8() {
    let mut embedding_output = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    embedding_output[0] = 4096;
    embedding_output[1] = 8192;
    let mut attention_context = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    attention_context[0] = 4096;
    attention_context[1] = -4096;
    let cache = MiniTransformerBlockForwardCache {
        block_input: embedding_output.clone(),
        attention_norm: embedding_output.clone(),
        attention_q: Vec::new(),
        attention_k: Vec::new(),
        attention_v: Vec::new(),
        attention_context,
        attention_probabilities_q15: Vec::new(),
        attention_output: Vec::new(),
        attention_residual: Vec::new(),
        mlp_norm: Vec::new(),
        mlp_up: Vec::new(),
        mlp_gate: Vec::new(),
        mlp_gated: Vec::new(),
        mlp_output: Vec::new(),
        block_output: Vec::new(),
        residual_saturation_count: 0,
    };
    let mut grad_q = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut grad_k = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut grad_v = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut grad_o = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    grad_q[0] = 1024;
    grad_k[1] = 1024;
    grad_v[0] = -1024;
    grad_o[0] = 1024;
    grad_o[1] = -1024;
    let mut gradient =
        MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL).expect("gradient");
    let mut scaled = [0_i32; MINI_TRANSFORMER_D_MODEL];

    accumulate_mini_transformer_attention_weight_gradient_i64(
        &cache,
        &grad_o,
        &grad_q,
        &grad_k,
        &grad_v,
        &mut gradient,
        &mut scaled,
    )
    .expect("first sample");
    accumulate_mini_transformer_attention_weight_gradient_i64(
        &cache,
        &grad_o,
        &grad_q,
        &grad_k,
        &grad_v,
        &mut gradient,
        &mut scaled,
    )
    .expect("second sample");

    let mut model = MiniTransformerMlpModel {
        context_seq_len: 1,
        embeddings: vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
        position_embeddings: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
        attention_rms_weights: Vec::new(),
        mlp_rms_weights: Vec::new(),
        q_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        k_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        v_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        o_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        up_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
        gate_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
        down_weights: vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
        output_weights: vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
    };
    let stats = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
        &mut gradient,
        &mut model,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 1,
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
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 12,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 22,
            attention_qk_learning_rate_shift: 22,
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
    .expect("apply");

    let row_1 = MINI_TRANSFORMER_D_MODEL;
    assert_eq!(&model.q_weights[..4], &[9, 8, 10, 10]);
    assert_eq!(&model.k_weights[row_1..row_1 + 4], &[9, 8, 10, 10]);
    assert_eq!(&model.v_weights[..4], &[11, 12, 10, 10]);
    assert_eq!(&model.o_weights[..4], &[9, 11, 10, 10]);
    assert_eq!(&model.o_weights[row_1..row_1 + 4], &[11, 9, 10, 10]);
    assert_eq!(stats.gradient_saturation_count, 0);
    assert_eq!(stats.zero_delta_count, 0);
    assert_eq!(stats.weight_delta_l1, 13);
    assert_eq!(gradient.q.sample_count, 0);
    assert_eq!(gradient.k.sample_count, 0);
    assert_eq!(gradient.v.sample_count, 0);
    assert_eq!(gradient.o.sample_count, 0);
}

#[test]
fn attention_qk_gradient_i64_carries_subthreshold_residuals() {
    let mut gradient =
        MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL).expect("gradient");
    let mut model = MiniTransformerMlpModel {
        context_seq_len: 1,
        embeddings: vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
        position_embeddings: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
        attention_rms_weights: Vec::new(),
        mlp_rms_weights: Vec::new(),
        q_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        k_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        v_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        o_weights: vec![10_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
        up_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
        gate_weights: vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM],
        down_weights: vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
        output_weights: vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL],
    };
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 1,
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
        learning_rate: 1,
        output_learning_rate_shift: 18,
        mlp_learning_rate_shift: 16,
        embedding_learning_rate_shift: 12,
        attention_learning_rate_shift: 22,
        attention_q_learning_rate_shift: 2,
        attention_qk_learning_rate_shift: 2,
        adaptive_rule_shifts: false,
        adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
        adaptive_attention_shifts: false,
        adaptive_holographic_shifts: false,
        attention_vo_error_feedback: false,
        attention_vo_oracle: false,
        reject_loss_regression: false,
        batch_mode: MiniTransformerBatchMode::Serial,
        map_reduce_workers: 1,
    };

    gradient.q.accumulators[0] = 1;
    gradient.q.sample_count = 1;
    gradient.k.accumulators[0] = 1;
    gradient.k.sample_count = 1;
    let first = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
        &mut gradient,
        &mut model,
        config,
    )
    .expect("first apply");

    assert_eq!(model.q_weights[0], 10);
    assert_eq!(model.k_weights[0], 10);
    assert_eq!(first.weight_delta_l1, 0);
    assert_eq!(first.zero_delta_count, 2);
    assert_eq!(gradient.q.residuals[0], 1);
    assert_eq!(gradient.k.residuals[0], 1);

    gradient.q.accumulators[0] = 1;
    gradient.q.sample_count = 1;
    gradient.k.accumulators[0] = 1;
    gradient.k.sample_count = 1;
    let second = apply_mini_transformer_attention_weight_gradient_i64_to_i8(
        &mut gradient,
        &mut model,
        config,
    )
    .expect("second apply");

    assert_eq!(model.q_weights[0], 9);
    assert_eq!(model.k_weights[0], 9);
    assert_eq!(second.weight_delta_l1, 2);
    assert_eq!(second.zero_delta_count, 0);
}

#[test]
fn embedding_gradient_i64_averages_then_updates_i16() {
    let context = [1_u8, 2_u8];
    let mut grad_embedding_output = vec![0_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
    grad_embedding_output[..4].copy_from_slice(&[4096, -4096, 0, 8192]);
    grad_embedding_output[MINI_TRANSFORMER_D_MODEL..MINI_TRANSFORMER_D_MODEL + 4]
        .copy_from_slice(&[-4096, 0, 4096, 0]);
    let mut gradient = MiniTransformerEmbeddingGradientI64::new(context.len()).expect("gradient");

    accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
        &context,
        &grad_embedding_output,
        MiniTransformerPositionPolicy::LearnedAbsolute,
        &mut gradient,
    )
    .expect("first sample");
    accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
        &context,
        &grad_embedding_output,
        MiniTransformerPositionPolicy::LearnedAbsolute,
        &mut gradient,
    )
    .expect("second sample");

    let mut embeddings = vec![10_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    let mut position_embeddings = vec![10_i16; context.len() * MINI_TRANSFORMER_D_MODEL];
    let stats = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
        &mut gradient,
        &mut embeddings,
        &mut position_embeddings,
        MiniTransformerPositionPolicy::LearnedAbsolute,
        1,
        12,
    )
    .expect("apply");

    let row_1 = usize::from(context[0]) * MINI_TRANSFORMER_D_MODEL;
    let row_2 = usize::from(context[1]) * MINI_TRANSFORMER_D_MODEL;
    let position_row_1 = 0;
    let position_row_2 = MINI_TRANSFORMER_D_MODEL;
    assert_eq!(&embeddings[row_1..row_1 + 4], &[9, 11, 10, 8]);
    assert_eq!(&embeddings[row_2..row_2 + 4], &[11, 10, 9, 10]);
    assert_eq!(
        &position_embeddings[position_row_1..position_row_1 + 4],
        &[9, 11, 10, 8]
    );
    assert_eq!(
        &position_embeddings[position_row_2..position_row_2 + 4],
        &[11, 10, 9, 10]
    );
    assert_eq!(stats.gradient_saturation_count, 0);
    assert_eq!(stats.zero_delta_count, 0);
    assert_eq!(stats.weight_delta_l1, 12);
    assert_eq!(gradient.sample_count, 0);
    assert!(gradient.token_accumulators.iter().all(|&value| value == 0));
    assert!(
        gradient
            .position_accumulators
            .iter()
            .all(|&value| value == 0)
    );
}

#[test]
fn embedding_gradient_i64_carries_subthreshold_residuals() {
    let mut gradient = MiniTransformerEmbeddingGradientI64::new(1).expect("gradient");
    let token = 7_u8;
    let context = [token];
    let mut grad_embedding_output = vec![0_i16; MINI_TRANSFORMER_D_MODEL];
    grad_embedding_output[0] = 1;
    let row = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
    let mut embeddings = vec![10_i16; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    let mut position_embeddings = Vec::new();

    accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
        &context,
        &grad_embedding_output,
        MiniTransformerPositionPolicy::Nope,
        &mut gradient,
    )
    .expect("first sample");
    let first = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
        &mut gradient,
        &mut embeddings,
        &mut position_embeddings,
        MiniTransformerPositionPolicy::Nope,
        1,
        2,
    )
    .expect("first apply");

    assert_eq!(embeddings[row], 10);
    assert_eq!(first.zero_delta_count, 1);
    assert_eq!(first.weight_delta_l1, 0);
    assert_eq!(gradient.residual_l1(MiniTransformerPositionPolicy::Nope), 1);

    accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
        &context,
        &grad_embedding_output,
        MiniTransformerPositionPolicy::Nope,
        &mut gradient,
    )
    .expect("second sample");
    let second = apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
        &mut gradient,
        &mut embeddings,
        &mut position_embeddings,
        MiniTransformerPositionPolicy::Nope,
        1,
        2,
    )
    .expect("second apply");

    assert_eq!(embeddings[row], 9);
    assert_eq!(second.zero_delta_count, 0);
    assert_eq!(second.weight_delta_l1, 1);
}
