//! Mini-transformer tests — gen.
use super::*;

#[test]
fn mini_transformer_model_round_trips_and_generates() {
    let tokens = b"To be or not to be, that is the question. To be or not to be. ";
    let run = run_mini_transformer_mlp_training_with_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(32),
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
    let bytes = run.model.to_bytes();
    let decoded = MiniTransformerMlpModel::from_bytes(&bytes).expect("model");

    assert_eq!(decoded, run.model);
    assert_eq!(decoded.model_hash(), run.trace.final_model_hash);
    assert_eq!(decoded.embedding_hash(), run.trace.final_embedding_hash);
    assert_eq!(decoded.attention_hash(), run.trace.final_attention_hash);
    assert_eq!(decoded.mlp_hash(), run.trace.final_mlp_hash);
    assert_eq!(decoded.output_head_hash(), run.trace.final_output_head_hash);

    let generation = generate_mini_transformer(&decoded, b"To be", ByteGenerationConfig::greedy(8))
        .expect("generate");

    assert_eq!(generation.generated_bytes.len(), 8);
    assert_eq!(generation.steps.len(), 8);
    assert_eq!(
        generation.attention_kind,
        MiniTransformerAttentionKind::Base2Softmax
    );
    assert_eq!(generation.context_seq_len, decoded.context_seq_len);
    assert_eq!(generation.model_hash, decoded.model_hash());
    assert_eq!(generation.attention_hash, decoded.attention_hash());
    assert_eq!(generation.mlp_hash, decoded.mlp_hash());
    assert_eq!(generation.output_head_hash, decoded.output_head_hash());
    let line = generation.to_json_line();
    assert!(line.contains("\"schema\":\"nsrl.mini_transformer_generation_trace.v1\""));
    assert!(line.contains("\"model\":\"mini_transformer_byte_qkvo_mlp_v1\""));
    assert!(line.contains("\"attention_kind\":\"base2_softmax\""));
}

#[test]
fn mini_transformer_generation_can_use_linear_attention() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(16);
    let generation = generate_mini_transformer_with_attention_kind(
        &model,
        b"To be",
        ByteGenerationConfig::greedy(4),
        MiniTransformerAttentionKind::Linear,
    )
    .expect("linear generation");

    assert_eq!(generation.generated_bytes.len(), 4);
    assert_eq!(generation.steps.len(), 4);
    assert_eq!(generation.context_seq_len, 16);
    assert_eq!(
        generation.attention_kind,
        MiniTransformerAttentionKind::Linear
    );
    let line = generation.to_json_line();
    assert!(line.contains("\"attention_kind\":\"linear\""));
}

#[test]
fn mini_transformer_generation_can_use_streaming_linear_attention_nope() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(16, 1)
        .expect("single-layer model");
    let generation = generate_mini_transformer_with_attention_kind(
        &model,
        b"To be",
        ByteGenerationConfig::greedy(4),
        MiniTransformerAttentionKind::LinearStreamingNope,
    )
    .expect("streaming linear generation");

    assert_eq!(generation.generated_bytes.len(), 4);
    assert_eq!(generation.steps.len(), 4);
    assert_eq!(generation.context_seq_len, 16);
    assert_eq!(
        generation.attention_kind,
        MiniTransformerAttentionKind::LinearStreamingNope
    );
    let line = generation.to_json_line();
    assert!(line.contains("\"attention_kind\":\"linear_streaming_nope\""));
    assert!(line.contains("\"position_policy\":\"nope\""));
    assert!(line.contains("\"incremental_attention_state\":true"));
    assert!(line.contains("\"streaming_nope_ignores_learned_position_embeddings\""));
    assert!(!line.contains("\"no_kv_cache_yet\""));
}

#[test]
fn mini_transformer_generation_can_use_streaming_linear_ttt_attention_nope() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(16, 1)
        .expect("single-layer model");
    let generation =
        generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
            &model,
            b"To be",
            ByteGenerationConfig::greedy(4),
            MiniTransformerAttentionKind::LinearStreamingTttNope,
            MiniTransformerPositionPolicy::Nope,
            None,
            DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT,
        )
        .expect("streaming linear ttt generation");

    assert_eq!(generation.generated_bytes.len(), 4);
    assert_eq!(generation.steps.len(), 4);
    assert_eq!(
        generation.attention_kind,
        MiniTransformerAttentionKind::LinearStreamingTttNope
    );
    let stats = generation.ttt_stats.expect("ttt stats");
    assert_eq!(
        stats.learning_rate_shift,
        DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT
    );
    assert_eq!(stats.step_count, b"To be".len() + 4);
    assert!(stats.total_state_delta_l1 > 0);
    let line = generation.to_json_line();
    assert!(line.contains("\"attention_kind\":\"linear_streaming_ttt_nope\""));
    assert!(line.contains("\"incremental_attention_state\":true"));
    assert!(line.contains("\"ttt\":{\"learning_rate_shift\":"));
}

#[test]
fn mini_transformer_generation_left_pads_short_prompt() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(16);
    let short =
        generate_mini_transformer(&model, b"a", ByteGenerationConfig::greedy(1)).expect("short");
    let mut padded_prompt = vec![b' '; 15];
    padded_prompt.push(b'a');
    let explicit =
        generate_mini_transformer(&model, &padded_prompt, ByteGenerationConfig::greedy(1))
            .expect("explicit");

    assert_eq!(short.context_seq_len, 16);
    assert_eq!(short.steps[0], explicit.steps[0]);
    assert_eq!(short.generated_bytes, explicit.generated_bytes);
}

#[test]
fn mini_transformer_eval_is_deterministic_and_read_only() {
    let tokens = b"crowley shakespeare blake literary evaluation fixture";
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
    let config = MiniTransformerMlpEvalConfig {
        seq_len: 8,
        stride: 3,
        max_windows: Some(7),
        attention_kind: MiniTransformerAttentionKind::Linear,
        position_policy: MiniTransformerPositionPolicy::Nope,
    };
    let before_hash = model.model_hash();
    let left = evaluate_mini_transformer_mlp_model(tokens, &model, config).expect("left");
    let right = evaluate_mini_transformer_mlp_model(tokens, &model, config).expect("right");

    assert_eq!(left, right);
    assert_eq!(left.windows, 7);
    assert_eq!(left.model_hash, before_hash);
    assert_eq!(model.model_hash(), before_hash);
    assert_eq!(left.invalid_forward_count, 0);
    assert!(left.unique_predicted_tokens > 0);
    assert!(left.unique_predicted_tokens <= BYTE_VOCAB);
    assert!(left.most_predicted_token.is_some());
    assert!(left.most_predicted_token_count <= left.windows);
    assert_eq!(
        left.most_predicted_token_share_per_mille,
        left.most_predicted_token_count * 1000 / left.windows
    );
    let json = left.to_json_line();
    assert!(json.contains(MINI_TRANSFORMER_EVAL_SCHEMA));
    assert!(json.contains("\"most_predicted_token_share_per_mille\":"));
}

#[test]
fn mini_transformer_block_expert_zero_identity_and_artifact_are_locked() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    let expert =
        MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 17).expect("block expert");
    let base = mini_transformer_next_token_row_with_attention_kind_position_policy(
        &model,
        b"Blak",
        MiniTransformerAttentionKind::Base2Softmax,
        MiniTransformerPositionPolicy::LearnedAbsolute,
    )
    .expect("base row");
    let adapted = mini_transformer_next_token_row_with_block_expert(
        &model,
        &expert,
        b"Blak",
        MiniTransformerAttentionKind::Base2Softmax,
        MiniTransformerPositionPolicy::LearnedAbsolute,
    )
    .expect("adapted row");
    assert_eq!(adapted, base);

    let bytes = expert.to_bytes();
    assert_eq!(
        MiniTransformerBlockLowRankExpert::from_bytes(&bytes).expect("decode"),
        expert
    );
    let mut corrupt = bytes;
    corrupt[64] ^= 1;
    assert!(MiniTransformerBlockLowRankExpert::from_bytes(&corrupt).is_err());
}

#[test]
fn mini_transformer_block_expert_training_updates_only_expert() {
    let tokens = b"Crowley Shakespeare Blake sing through the integer swarm.";
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    let model_hash = model.model_hash();
    let mut expert =
        MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 23).expect("block expert");
    let stats = train_mini_transformer_block_expert(
        tokens,
        &model,
        &mut expert,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            max_windows: Some(4),
            batch_windows: 2,
            ..MiniTransformerMlpTrainConfig::default()
        },
        2,
        1024,
        0,
    )
    .expect("train block expert");
    assert_eq!(model.model_hash(), model_hash);
    assert_eq!(stats.optimizer_steps, 2);
    assert!(stats.weight_delta_l1 > 0);
    assert!(
        expert
            .expansion_weights_q15
            .iter()
            .any(|&weight| weight != 0)
    );
}

#[test]
fn mini_transformer_block_expert_raw_probability_gradient_and_guard_are_locked() {
    let tokens = b"Crowley Shakespeare Blake sing through the integer swarm.";
    let model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 2)
        .expect("two-layer model");
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        max_windows: Some(4),
        batch_windows: 4,
        ..MiniTransformerMlpTrainConfig::default()
    };

    let mut unguarded =
        MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 29).expect("expert");
    let unguarded_stats = train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
        tokens,
        &model,
        &mut unguarded,
        config,
        4,
        1024,
        0,
        Some(1),
        false,
        MiniTransformerBlockExpertObjective::ProbabilityError,
    )
    .expect("metric-aligned update");
    assert_eq!(unguarded_stats.optimizer_steps, 1);
    assert!(unguarded_stats.weight_delta_l1 > 0);
    let parameters_per_layer = MINI_TRANSFORMER_D_MODEL * unguarded.rank;
    assert!(
        unguarded.expansion_weights_q15[..parameters_per_layer]
            .iter()
            .all(|&weight| weight == 0)
    );
    assert!(
        unguarded.expansion_weights_q15[parameters_per_layer..]
            .iter()
            .any(|&weight| weight != 0)
    );

    let mut guarded =
        MiniTransformerBlockLowRankExpert::new_for_model(&model, 4, 29).expect("expert");
    let baseline = evaluate_mini_transformer_block_expert(
        tokens,
        &model,
        &guarded,
        MiniTransformerMlpEvalConfig {
            seq_len: 4,
            stride: 1,
            max_windows: Some(4),
            attention_kind: config.attention_kind,
            position_policy: config.position_policy,
        },
    )
    .expect("baseline");
    let guarded_stats = train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
        tokens,
        &model,
        &mut guarded,
        config,
        4,
        1024,
        0,
        Some(1),
        true,
        MiniTransformerBlockExpertObjective::ProbabilityError,
    )
    .expect("guarded update");
    let final_metrics = evaluate_mini_transformer_block_expert(
        tokens,
        &model,
        &guarded,
        MiniTransformerMlpEvalConfig {
            seq_len: 4,
            stride: 1,
            max_windows: Some(4),
            attention_kind: config.attention_kind,
            position_policy: config.position_policy,
        },
    )
    .expect("final metrics");
    assert!(final_metrics.probability_error_q15 <= baseline.probability_error_q15);
    assert_eq!(
        guarded_stats.accepted_forward_steps
            + guarded_stats.accepted_reverse_steps
            + guarded_stats.rejected_steps,
        guarded_stats.optimizer_steps
    );
    assert!(
        guarded.expansion_weights_q15[..parameters_per_layer]
            .iter()
            .all(|&weight| weight == 0)
    );
}
