//! Mini-transformer tests — trace.
use super::*;

#[test]
fn mini_transformer_mlp_training_trace_is_byte_stable() {
    let tokens = b"abababababab";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(6),
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
    };
    let left = run_mini_transformer_mlp_training(tokens, config)
        .expect("left")
        .to_json_line();
    let right = run_mini_transformer_mlp_training(tokens, config)
        .expect("right")
        .to_json_line();

    assert_eq!(left, right);
    assert!(left.contains("\"schema\":\"nsrl.training_mini_transformer_mlp_trace.v1\""));
    assert!(left.contains("\"attention\":\"updates_q_k_v_o_i8\""));
    assert!(left.contains("\"rejected_window_count\":"));
    assert!(left.contains("\"final_invalid_forward_count\":"));
    assert!(left.contains(
            "\"trained_component\":\"embedding_i16_plus_output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8\""
        ));
}

#[test]
fn mini_transformer_binary_trace_has_fixed_step_records() {
    let tokens = b"abababababab";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(6),
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
    };
    let trace = run_mini_transformer_mlp_training(tokens, config).expect("trace");
    let binary = trace.to_binary_trace_v1();
    let final_offset = 16 + trace.steps.len() * 32;

    assert_eq!(&binary[..4], MINI_TRANSFORMER_BINARY_TRACE_MAGIC);
    assert_eq!(binary[4], MINI_TRANSFORMER_BINARY_TRACE_VERSION);
    assert_eq!(binary[5], MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID);
    assert_eq!(binary[16], MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE);
    assert_eq!(
        binary[final_offset],
        MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY
    );
    assert_eq!(binary[final_offset + 1], 0);
}

#[test]
fn mini_transformer_streamed_binary_trace_matches_buffered_trace() {
    let tokens = b"to be or not to be ";
    let config = MiniTransformerMlpTrainConfig {
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
    };
    let mut streamed = Vec::new();
    let buffered = {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        let mut writer = MiniTransformerBinaryTraceWriter::new(&mut streamed);
        let run =
                run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace(
                    tokens,
                    config,
                    model,
                    0,
                    MiniTransformerTraceDetail::Summary,
                    |_| Ok(()),
                    |record| writer.write_record(record).map_err(|_| TrainError::TraceWrite),
                )
                .expect("streamed binary trace");
        run.trace.to_binary_trace_v1()
    };

    assert_eq!(streamed, buffered);
}

#[test]
fn mini_transformer_swarm_trains_interleaved_worker_shards() {
    let tokens = b"to be or not to be to be or not to be ";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(8),
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
    };
    let run = run_mini_transformer_mlp_swarm_training(
        tokens,
        config,
        MiniTransformerMlpSwarmTrainConfig {
            workers: 2,
            trace_detail: MiniTransformerTraceDetail::None,
        },
    )
    .expect("swarm training");

    assert_eq!(run.trace.worker_count, 2);
    assert_eq!(run.trace.workers.len(), 2);
    assert_eq!(run.trace.workers[0].window_offset, 0);
    assert_eq!(run.trace.workers[1].window_offset, 1);
    assert_eq!(run.trace.workers[0].stride, 2);
    assert_eq!(run.trace.workers[1].stride, 2);
    assert_eq!(run.trace.workers[0].max_windows, Some(4));
    assert_eq!(run.trace.workers[1].max_windows, Some(4));
    assert_eq!(run.trace.final_model_hash, run.model.model_hash());
    assert!(
        run.trace
            .workers
            .iter()
            .any(|worker| worker.worker_index == run.trace.best_worker_index)
    );
    assert!(
        run.trace
            .to_json_line()
            .contains("\"schema\":\"nsrl.training_mini_transformer_swarm_trace.v1\"")
    );
}

#[test]
fn mini_transformer_swarm_worker_artifacts_assemble_to_local_swarm() {
    let tokens = b"to be or not to be to be or not to be ";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(8),
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
    };
    let base_model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    let local = run_mini_transformer_mlp_swarm_training_from_model(
        tokens,
        config,
        MiniTransformerMlpSwarmTrainConfig {
            workers: 2,
            trace_detail: MiniTransformerTraceDetail::None,
        },
        base_model.clone(),
    )
    .expect("local swarm");
    let artifacts = (0..2)
        .map(|worker_index| {
            let run = run_mini_transformer_mlp_swarm_worker_from_model_with_progress(
                tokens,
                config,
                worker_index,
                2,
                base_model.clone(),
                0,
                MiniTransformerTraceDetail::None,
                |_| Ok(()),
            )
            .expect("worker");
            let bytes = run.artifact.try_to_bytes().expect("worker bytes");
            MiniTransformerMlpSwarmWorkerArtifact::from_bytes(&bytes).expect("worker artifact")
        })
        .collect::<Vec<_>>();
    let assembled = assemble_mini_transformer_mlp_swarm_worker_artifacts(
        tokens,
        config,
        &base_model,
        artifacts,
    )
    .expect("assembled swarm");

    assert_eq!(assembled.trace, local.trace);
    assert_eq!(assembled.model, local.model);
    assert_eq!(assembled.swarm_model, local.swarm_model);
    assert!(
        assembled
            .trace
            .to_json_line()
            .contains("\"schema\":\"nsrl.training_mini_transformer_swarm_trace.v1\"")
    );
}

#[test]
fn mini_transformer_swarm_trace_is_byte_stable() {
    let tokens = b"abababababababab";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(6),
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
    };
    let swarm_config = MiniTransformerMlpSwarmTrainConfig {
        workers: 3,
        trace_detail: MiniTransformerTraceDetail::None,
    };
    let left = run_mini_transformer_mlp_swarm_training(tokens, config, swarm_config)
        .expect("left")
        .trace
        .to_json_line();
    let right = run_mini_transformer_mlp_swarm_training(tokens, config, swarm_config)
        .expect("right")
        .trace
        .to_json_line();

    assert_eq!(left, right);
}

#[test]
fn mini_transformer_swarm_scaling_benchmark_sweeps_worker_counts() {
    let tokens = b"abababababababab";
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(6),
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
    };
    let trace = run_mini_transformer_mlp_swarm_scaling_benchmark(
        tokens,
        config,
        3,
        MiniTransformerTraceDetail::None,
    )
    .expect("scaling benchmark");

    assert_eq!(trace.worker_counts, vec![1, 2, 3]);
    assert_eq!(trace.runs.len(), 3);
    assert_eq!(trace.runs[0].requested_worker_count, 1);
    assert_eq!(trace.runs[0].effective_worker_count, 1);
    assert_eq!(trace.runs[0].speedup_per_mille, 1000);
    assert!(trace.runs.iter().all(|run| {
        run.effective_worker_count > 0
            && run.effective_worker_count <= run.requested_worker_count
            && run.examined_windows > 0
    }));

    let json = trace.to_json_line();
    assert!(json.contains("\"schema\":\"nsrl.training_mini_transformer_swarm_scaling_trace.v1\""));
    assert!(json.contains("\"worker_counts\":[1,2,3]"));
    assert!(json.contains("\"speedup_per_mille\""));
}
