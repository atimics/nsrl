//! Mini-transformer tests — adam.
use super::*;

fn tiny_integer_adam_training_config() -> MiniTransformerMlpTrainConfig {
    MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        max_windows: Some(8),
        batch_windows: 4,
        attention_kind: MiniTransformerAttentionKind::Linear,
        position_policy: MiniTransformerPositionPolicy::Nope,
        batch_mode: MiniTransformerBatchMode::Serial,
        ..MiniTransformerMlpTrainConfig::default()
    }
}

#[test]
fn mini_transformer_integer_adam_replay_is_deterministic() {
    let tokens = b"to be or not to be, that is the question";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };

    let left = run_mini_transformer_mlp_integer_adam_training(tokens, config, optimizer)
        .expect("left Adam run");
    let right = run_mini_transformer_mlp_integer_adam_training(tokens, config, optimizer)
        .expect("right Adam run");

    assert_eq!(left, right);
    assert_eq!(left.trace.updates, 8);
    assert_eq!(left.trace.optimizer_step, 2);
    assert_ne!(left.trace.initial_model_hash, left.trace.final_model_hash);
    assert!(left.trace.output_head_delta_l1 > 0);
    assert!(left.trace.mlp_delta_l1 > 0);
    assert!(left.trace.embedding_delta_l1 > 0);
    assert!(left.trace.attention_delta_l1 > 0);
    assert!(left.trace.attention_q_delta_l1 > 0);
    assert!(left.trace.attention_k_delta_l1 > 0);
    assert!(left.trace.attention_v_delta_l1 > 0);
    assert!(left.trace.attention_o_delta_l1 > 0);
    assert_eq!(left.trace.transformer_layers, 2);
    let json = left.trace.to_json_line();
    assert!(json.contains("\"architecture_profile\""));
    assert!(json.contains("\"argmax_margin_weight_q15\":0"));
    assert!(json.contains("\"attention_q\":"));
    assert!(json.contains("\"saturation\":"));
    left.optimizer_state
        .validate_for_model(&left.model)
        .expect("final optimizer binding");
}

#[cfg(feature = "mini-calibrated")]
#[test]
fn calibrated_suffix_memory_is_deterministic_and_uses_longest_suffix() {
    let tokens = b"abcaabdxabcaabdy";
    let mut first = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    let mut replay = first.clone();

    let first_records =
        mini_transformer_install_ngram_cache(&mut first, tokens).expect("first cache");
    let replay_records =
        mini_transformer_install_ngram_cache(&mut replay, tokens).expect("replayed cache");

    assert_eq!(first_records, replay_records);
    assert_eq!(first.position_embeddings, replay.position_embeddings);
    assert_eq!(
        mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"zzabc"),
        Some(b'a')
    );
    assert_eq!(
        mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"zzabd"),
        Some(b'x')
    );
    assert_eq!(
        mini_transformer_ngram_cache_prediction(&first.position_embeddings, b"qqqq"),
        Some(b'a')
    );
}

#[test]
fn mini_transformer_integer_adam_updates_rmsnorm_gamma() {
    let tokens = b"Every thing that lives is holy, and each breath returns.";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    model.enable_rms_norm().expect("enable RMSNorm");
    let initial_attention_gamma = model.attention_rms_weights.clone();
    let initial_mlp_gamma = model.mlp_rms_weights.clone();

    let run = run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens, config, optimizer, model, None,
    )
    .expect("RMSNorm Adam run");

    assert!(run.model.rms_norm_enabled());
    assert_ne!(run.model.attention_rms_weights, initial_attention_gamma);
    assert_ne!(run.model.mlp_rms_weights, initial_mlp_gamma);
    assert!(run.trace.rms_norm_delta_l1 > 0);
    run.optimizer_state
        .validate_for_model(&run.model)
        .expect("RMS optimizer binding");
}

#[test]
fn mini_transformer_rmsnorm_scope_updates_only_internal_gamma() {
    let tokens = b"The imagination is not a state; it is existence itself.";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    model.enable_rms_norm().expect("enable RMSNorm");
    let initial = model.clone();

    let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer,
        model,
        None,
        MiniTransformerAdamTrainScope::RmsNorm,
    )
    .expect("RMSNorm-only Adam run");

    assert_eq!(
        run.trace.train_scope,
        MiniTransformerAdamTrainScope::RmsNorm
    );
    assert!(run.trace.rms_norm_delta_l1 > 0);
    assert_eq!(run.trace.output_head_delta_l1, 0);
    assert_eq!(run.trace.mlp_delta_l1, 0);
    assert_eq!(run.trace.embedding_delta_l1, 0);
    assert_eq!(run.trace.attention_delta_l1, 0);
    assert_ne!(
        run.model.attention_rms_weights,
        initial.attention_rms_weights
    );
    assert_ne!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
    assert_eq!(run.model.embeddings, initial.embeddings);
    assert_eq!(run.model.position_embeddings, initial.position_embeddings);
    assert_eq!(run.model.q_weights, initial.q_weights);
    assert_eq!(run.model.k_weights, initial.k_weights);
    assert_eq!(run.model.v_weights, initial.v_weights);
    assert_eq!(run.model.o_weights, initial.o_weights);
    assert_eq!(run.model.up_weights, initial.up_weights);
    assert_eq!(run.model.gate_weights, initial.gate_weights);
    assert_eq!(run.model.down_weights, initial.down_weights);
    assert_eq!(run.model.output_weights, initial.output_weights);
}

#[test]
fn mini_transformer_final_mlp_scope_freezes_shared_trunk() {
    let tokens = b"The tygers of wrath are wiser than the horses of instruction.";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    model.enable_rms_norm().expect("enable RMSNorm");
    let initial = model.clone();
    let layers = model.transformer_layers();
    assert!(layers > 1);
    let final_up = model
        .mlp_up_or_gate_weight_range(layers - 1)
        .expect("final up range");
    let final_down = model
        .mlp_down_weight_range(layers - 1)
        .expect("final down range");

    let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer,
        model,
        None,
        MiniTransformerAdamTrainScope::FinalMlp,
    )
    .expect("final MLP Adam run");

    assert_eq!(
        run.trace.train_scope,
        MiniTransformerAdamTrainScope::FinalMlp
    );
    assert_eq!(run.trace.output_head_delta_l1, 0);
    assert_eq!(run.trace.embedding_delta_l1, 0);
    assert_eq!(run.trace.rms_norm_delta_l1, 0);
    assert_eq!(run.trace.attention_delta_l1, 0);
    assert!(run.trace.mlp_delta_l1 > 0);
    assert_eq!(run.model.embeddings, initial.embeddings);
    assert_eq!(run.model.position_embeddings, initial.position_embeddings);
    assert_eq!(
        run.model.attention_rms_weights,
        initial.attention_rms_weights
    );
    assert_eq!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
    assert_eq!(run.model.q_weights, initial.q_weights);
    assert_eq!(run.model.k_weights, initial.k_weights);
    assert_eq!(run.model.v_weights, initial.v_weights);
    assert_eq!(run.model.o_weights, initial.o_weights);
    assert_eq!(run.model.output_weights, initial.output_weights);
    assert_eq!(
        run.model.up_weights[..final_up.start],
        initial.up_weights[..final_up.start]
    );
    assert_eq!(
        run.model.gate_weights[..final_up.start],
        initial.gate_weights[..final_up.start]
    );
    assert_eq!(
        run.model.down_weights[..final_down.start],
        initial.down_weights[..final_down.start]
    );
    assert_ne!(
        run.model.up_weights[final_up.clone()],
        initial.up_weights[final_up]
    );
    assert_ne!(
        run.model.down_weights[final_down.clone()],
        initial.down_weights[final_down]
    );
    run.optimizer_state
        .validate_for_model(&run.model)
        .expect("final MLP optimizer binding");
}

#[test]
fn mini_transformer_final_mlp_output_scope_updates_expert_only() {
    let tokens = b"Energy is eternal delight, carried by the speaking flame.";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    model.enable_rms_norm().expect("enable RMSNorm");
    let initial = model.clone();

    let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer,
        model,
        None,
        MiniTransformerAdamTrainScope::FinalMlpAndOutput,
    )
    .expect("final MLP and output Adam run");

    assert_eq!(
        run.trace.train_scope,
        MiniTransformerAdamTrainScope::FinalMlpAndOutput
    );
    assert!(run.trace.output_head_delta_l1 > 0);
    assert!(run.trace.mlp_delta_l1 > 0);
    assert_eq!(run.trace.embedding_delta_l1, 0);
    assert_eq!(run.trace.rms_norm_delta_l1, 0);
    assert_eq!(run.trace.attention_delta_l1, 0);
    assert_eq!(run.model.embeddings, initial.embeddings);
    assert_eq!(run.model.position_embeddings, initial.position_embeddings);
    assert_eq!(
        run.model.attention_rms_weights,
        initial.attention_rms_weights
    );
    assert_eq!(run.model.mlp_rms_weights, initial.mlp_rms_weights);
    assert_eq!(run.model.q_weights, initial.q_weights);
    assert_eq!(run.model.k_weights, initial.k_weights);
    assert_eq!(run.model.v_weights, initial.v_weights);
    assert_eq!(run.model.o_weights, initial.o_weights);
    assert_ne!(run.model.output_weights, initial.output_weights);
}

#[test]
fn mini_transformer_output_scope_freezes_all_hidden_layers() {
    let tokens = b"All deities reside in the human breast.";
    let config = tiny_integer_adam_training_config();
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    let initial = model.clone();
    let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer,
        model,
        None,
        MiniTransformerAdamTrainScope::Output,
    )
    .expect("output-only Adam run");

    assert_eq!(run.trace.train_scope, MiniTransformerAdamTrainScope::Output);
    assert!(run.trace.output_head_delta_l1 > 0);
    assert_eq!(run.trace.mlp_delta_l1, 0);
    assert_eq!(run.trace.embedding_delta_l1, 0);
    assert_eq!(run.trace.rms_norm_delta_l1, 0);
    assert_eq!(run.trace.attention_delta_l1, 0);
    assert_eq!(run.model.embeddings, initial.embeddings);
    assert_eq!(run.model.q_weights, initial.q_weights);
    assert_eq!(run.model.up_weights, initial.up_weights);
    assert_ne!(run.model.output_weights, initial.output_weights);
}

#[test]
fn mini_transformer_rmsnorm_adam_serial_map_reduce_parity() {
    let tokens = b"The road of excess leads to the palace of wisdom.";
    let serial_config = MiniTransformerMlpTrainConfig {
        max_windows: Some(4),
        ..tiny_integer_adam_training_config()
    };
    let map_reduce_config = MiniTransformerMlpTrainConfig {
        batch_mode: MiniTransformerBatchMode::MapReduce,
        map_reduce_workers: 2,
        ..serial_config
    };
    let optimizer = IntegerAdamConfig {
        step_shift: 0,
        ..IntegerAdamConfig::default()
    };
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(serial_config.seq_len);
    model.enable_rms_norm().expect("enable RMSNorm");
    let serial = run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens,
        serial_config,
        optimizer,
        model.clone(),
        None,
    )
    .expect("serial RMSNorm run");
    let map_reduce = run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens,
        map_reduce_config,
        optimizer,
        model,
        None,
    )
    .expect("map-reduce RMSNorm run");

    assert_eq!(serial.model, map_reduce.model);
    assert_eq!(serial.optimizer_state, map_reduce.optimizer_state);
    assert_eq!(
        serial.trace.rms_norm_delta_l1,
        map_reduce.trace.rms_norm_delta_l1
    );
}

#[test]
fn mini_transformer_integer_adam_serial_map_reduce_parity() {
    let tokens = b"Tyger Tyger, burning bright, in the forests of the night";
    let serial_config = tiny_integer_adam_training_config();
    let map_reduce_config = MiniTransformerMlpTrainConfig {
        batch_mode: MiniTransformerBatchMode::MapReduce,
        map_reduce_workers: 2,
        ..serial_config
    };
    let optimizer = IntegerAdamConfig {
        step_shift: 1,
        ..IntegerAdamConfig::default()
    };

    let serial = run_mini_transformer_mlp_integer_adam_training(tokens, serial_config, optimizer)
        .expect("serial Adam run");
    let map_reduce =
        run_mini_transformer_mlp_integer_adam_training(tokens, map_reduce_config, optimizer)
            .expect("map-reduce Adam run");

    assert_eq!(serial.model, map_reduce.model);
    assert_eq!(serial.optimizer_state, map_reduce.optimizer_state);
    assert_eq!(
        serial.trace.final_model_hash,
        map_reduce.trace.final_model_hash
    );
    assert_eq!(
        serial.trace.optimizer_state_hash,
        map_reduce.trace.optimizer_state_hash
    );
}

#[test]
fn mini_transformer_integer_adam_resume_matches_uninterrupted_training() {
    let tokens = b"Shall I compare thee to a summer's day? Thou art more lovely.";
    let one_epoch = tiny_integer_adam_training_config();
    let two_epochs = MiniTransformerMlpTrainConfig {
        epochs: 2,
        ..one_epoch
    };
    let optimizer = IntegerAdamConfig {
        step_shift: 1,
        ..IntegerAdamConfig::default()
    };
    let uninterrupted =
        run_mini_transformer_mlp_integer_adam_training(tokens, two_epochs, optimizer)
            .expect("uninterrupted Adam run");
    let first = run_mini_transformer_mlp_integer_adam_training(tokens, one_epoch, optimizer)
        .expect("first resumed epoch");
    let state_bytes = first.optimizer_state.to_bytes();
    let resumed_state =
        MiniTransformerAdamOptimizerState::from_bytes(&state_bytes).expect("resume state decode");
    let resumed = run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens,
        one_epoch,
        optimizer,
        MiniTransformerMlpModel::from_bytes(&first.model.to_bytes()).expect("resume model decode"),
        Some(resumed_state),
    )
    .expect("resumed Adam run");

    assert_eq!(uninterrupted.model, resumed.model);
    assert_eq!(uninterrupted.optimizer_state, resumed.optimizer_state);
    assert_eq!(
        uninterrupted.trace.final_model_hash,
        resumed.trace.final_model_hash
    );
    assert_eq!(
        uninterrupted.trace.optimizer_step,
        resumed.trace.optimizer_step
    );
}
