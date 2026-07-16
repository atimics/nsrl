//! Mini-transformer unit, replay, parity, and artifact regression tests.

use super::*;
#[test]
fn checked_model_serialization_rejects_oversized_public_shapes() {
    if usize::BITS <= 32 {
        return;
    }

    let too_large = u32::MAX as usize + 1;

    let mini = MiniTransformerMlpModel {
        context_seq_len: too_large,
        embeddings: Vec::new(),
        position_embeddings: Vec::new(),
        attention_rms_weights: Vec::new(),
        mlp_rms_weights: Vec::new(),
        q_weights: Vec::new(),
        k_weights: Vec::new(),
        v_weights: Vec::new(),
        o_weights: Vec::new(),
        up_weights: Vec::new(),
        gate_weights: Vec::new(),
        down_weights: Vec::new(),
        output_weights: Vec::new(),
    };
    assert!(mini.try_to_bytes().is_err());
}

#[test]
fn mini_transformer_adam_state_round_trips_separately_from_model() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
    let mut state =
        MiniTransformerAdamOptimizerState::new_for_model(&model, IntegerAdamConfig::default())
            .expect("new Adam state");
    state.step = 17;
    let last_parameter = state.parameter_count() - 1;
    state.first_moments[0] = -123;
    state.first_moments[last_parameter] = 456;
    state.second_moments[1] = 789;
    state.update_residuals[2] = -321;

    let bytes = state.to_bytes();
    let decoded = MiniTransformerAdamOptimizerState::from_bytes(&bytes).expect("decode Adam state");

    assert_eq!(decoded, state);
    decoded
        .validate_for_model(&model)
        .expect("state remains bound to model");
    assert_eq!(&bytes[..8], MINI_TRANSFORMER_ADAM_STATE_MAGIC);
    assert_ne!(&bytes[..8], MINI_TRANSFORMER_MODEL_MAGIC);
}

#[test]
fn mini_transformer_adam_state_rejects_corruption_and_wrong_model() {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    let state =
        MiniTransformerAdamOptimizerState::new_for_model(&model, IntegerAdamConfig::default())
            .expect("new Adam state");
    let mut corrupt = state.to_bytes();
    corrupt[80] ^= 0x40;
    assert!(MiniTransformerAdamOptimizerState::from_bytes(&corrupt).is_err());
    assert!(MiniTransformerAdamOptimizerState::from_bytes(&corrupt[..64]).is_err());

    let mut changed_model = model.clone();
    changed_model.output_weights[0] = changed_model.output_weights[0].saturating_add(1);
    assert!(state.validate_for_model(&changed_model).is_err());

    let mut rebound = state.clone();
    rebound
        .bind_to_model(&changed_model)
        .expect("same-shape model can receive state after an accepted update");
    rebound
        .validate_for_model(&changed_model)
        .expect("rebound state/model pair");
}

#[test]
fn mini_transformer_rms_model_round_trips_and_same_geometry_v4_stays_disabled() {
    let mut rms_model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
    rms_model.enable_rms_norm().expect("enable RMSNorm");
    assert!(rms_model.rms_norm_enabled());
    let rms_bytes = rms_model.to_bytes();
    assert_eq!(&rms_bytes[..8], MINI_TRANSFORMER_MODEL_MAGIC);
    let decoded = MiniTransformerMlpModel::from_bytes(&rms_bytes).expect("RMS model decode");
    assert_eq!(decoded, rms_model);

    let legacy_model = MiniTransformerMlpModel::new_initial_with_seq_len(8);
    let mut legacy_bytes = legacy_model.to_bytes();
    legacy_bytes[..8].copy_from_slice(MINI_TRANSFORMER_LEGACY_MODEL_MAGIC);
    let decoded_legacy =
        MiniTransformerMlpModel::from_bytes(&legacy_bytes).expect("legacy v4 decode");
    assert_eq!(decoded_legacy, legacy_model);
    assert!(!decoded_legacy.rms_norm_enabled());
}

fn historical_v4_fixture_bytes(context_seq_len: usize) -> Vec<u8> {
    let embeddings = vec![0_i16; BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
    let position_embeddings = vec![0_i16; context_seq_len * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
    let attention_count = MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_D_MODEL;
    let mut q_weights = vec![0_i8; attention_count];
    let mut k_weights = vec![0_i8; attention_count];
    let mut v_weights = vec![0_i8; attention_count];
    let mut o_weights = vec![0_i8; attention_count];
    for index in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        let diagonal = index * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + index;
        q_weights[diagonal] = 1;
        k_weights[diagonal] = 1;
        v_weights[diagonal] = 1;
        o_weights[diagonal] = 1;
    }
    let up_weights =
        vec![0_i8; MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];
    let gate_weights = up_weights.clone();
    let down_weights =
        vec![0_i8; MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM];
    let output_weights = vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL];

    let mut embedding_hasher = StableHasher::new();
    embedding_hasher.update_i16_slice(&embeddings);
    embedding_hasher.update_i16_slice(&position_embeddings);
    let mut model_hasher = StableHasher::new();
    model_hasher.update_usize(context_seq_len);
    model_hasher.update_i16_slice(&embeddings);
    model_hasher.update_i16_slice(&position_embeddings);
    model_hasher.update_i8_slice(&q_weights);
    model_hasher.update_i8_slice(&k_weights);
    model_hasher.update_i8_slice(&v_weights);
    model_hasher.update_i8_slice(&o_weights);
    model_hasher.update_i8_slice(&up_weights);
    model_hasher.update_i8_slice(&gate_weights);
    model_hasher.update_i8_slice(&down_weights);
    model_hasher.update_i8_slice(&output_weights);

    let tensors = [
        q_weights.as_slice(),
        k_weights.as_slice(),
        v_weights.as_slice(),
        o_weights.as_slice(),
        up_weights.as_slice(),
        gate_weights.as_slice(),
        down_weights.as_slice(),
        output_weights.as_slice(),
    ];
    let counts = [
        embeddings.len(),
        position_embeddings.len(),
        q_weights.len(),
        k_weights.len(),
        v_weights.len(),
        o_weights.len(),
        up_weights.len(),
        gate_weights.len(),
        down_weights.len(),
        output_weights.len(),
    ];
    let hashes = [
        embedding_hasher.finish(),
        hash_i8_slice(&q_weights),
        hash_i8_slice(&k_weights),
        hash_i8_slice(&v_weights),
        hash_i8_slice(&o_weights),
        hash_three_i8_slices(&up_weights, &gate_weights, &down_weights),
        hash_i8_slice(&output_weights),
        model_hasher.finish(),
    ];
    let mut out = Vec::new();
    out.extend_from_slice(MINI_TRANSFORMER_LEGACY_MODEL_MAGIC);
    for value in [
        BYTE_VOCAB,
        MINI_TRANSFORMER_LEGACY_V4_D_MODEL,
        MINI_TRANSFORMER_LEGACY_V4_HEADS,
        MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM,
        context_seq_len,
    ] {
        out.extend_from_slice(&(value as u32).to_le_bytes());
    }
    for count in counts {
        out.extend_from_slice(&(count as u64).to_le_bytes());
    }
    for hash in hashes {
        out.extend_from_slice(&hash.to_le_bytes());
    }
    for value in embeddings.iter().chain(position_embeddings.iter()) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for tensor in tensors {
        out.extend(tensor.iter().map(|&value| value as u8));
    }
    out
}

#[test]
fn historical_v4_geometry_upgrades_for_eval_and_resume() {
    let bytes = historical_v4_fixture_bytes(4);
    let source_hash =
        MiniTransformerMlpModel::serialized_model_hash(&bytes).expect("serialized source hash");
    let upgraded = MiniTransformerMlpModel::from_bytes(&bytes).expect("historical V4 decode");
    assert_eq!(upgraded.context_seq_len, 4);
    assert_eq!(
        upgraded.embeddings.len(),
        BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
    );
    assert_eq!(upgraded.transformer_layers(), 1);
    assert_ne!(upgraded.model_hash(), source_hash);
    assert_eq!(&upgraded.to_bytes()[..8], MINI_TRANSFORMER_MODEL_MAGIC);

    evaluate_mini_transformer_mlp_model(
        b"legacy checkpoint evaluation",
        &upgraded,
        MiniTransformerMlpEvalConfig {
            seq_len: 4,
            stride: 1,
            max_windows: Some(1),
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
        },
    )
    .expect("upgraded model evaluates");

    run_mini_transformer_mlp_training_from_model(
        b"legacy checkpoint resume",
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            max_windows: Some(1),
            batch_windows: 1,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            ..MiniTransformerMlpTrainConfig::default()
        },
        upgraded,
    )
    .expect("upgraded model resumes training");
}

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

#[test]
fn byte_target_frequency_weights_only_downweight_common_targets() {
    let tokens = [b'x', b'a', b'y', b'a', b'z', b'a', b'w', b'b'];
    let weights = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 2, 4096)
        .expect("byte target frequency weights");

    assert!(weights[usize::from(b'a')] < i16::MAX);
    assert!(weights[usize::from(b'a')] >= 4096);
    assert_eq!(weights[usize::from(b'b')], i16::MAX);
    assert_eq!(weights[usize::from(b'c')], i16::MAX);

    let disabled = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 0, 4096)
        .expect("disabled byte target frequency weights");
    assert!(disabled.iter().all(|&weight| weight == i16::MAX));
}

#[test]
fn byte_argmax_margin_gradient_pushes_target_against_best_competitor() {
    let mut gradient = [0_i32; BYTE_VOCAB];
    let mut logits = [0_i32; BYTE_VOCAB];
    logits[usize::from(b'a')] = 10;
    logits[usize::from(b'b')] = 12;
    logits[usize::from(b'c')] = 12;

    apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);

    assert!(gradient[usize::from(b'a')] < 0);
    assert!(gradient[usize::from(b'b')] > 0);
    assert_eq!(gradient[usize::from(b'c')], 0);

    let pushed_target = gradient[usize::from(b'a')];
    let pushed_competitor = gradient[usize::from(b'b')];
    logits[usize::from(b'a')] = 13;
    apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);
    assert_eq!(gradient[usize::from(b'a')], pushed_target);
    assert_eq!(gradient[usize::from(b'b')], pushed_competitor);
}

#[test]
fn sample_decode_is_deterministic_and_can_escape_argmax() {
    let logits = [0_i32; BYTE_VOCAB];
    let mut probabilities = [0_i16; BYTE_VOCAB];
    for probability in probabilities.iter_mut().take(4) {
        *probability = 8192;
    }
    let decode = DecodeConfig {
        strategy: DecodeStrategy::Sample,
        sample_seed: 7,
        top_k: 4,
        ..DecodeConfig::greedy()
    };

    let left =
        select_byte_from_row(&logits, &probabilities, decode, 3, b"context").expect("sample left");
    let right =
        select_byte_from_row(&logits, &probabilities, decode, 3, b"context").expect("sample right");

    assert_eq!(left, right);
    assert!(left < 4);
    assert!((0..64).any(|seed| {
        let decode = DecodeConfig {
            strategy: DecodeStrategy::Sample,
            sample_seed: seed,
            top_k: 4,
            ..DecodeConfig::greedy()
        };
        select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
            .is_ok_and(|token| token != 0 && token < 4)
    }));
}

#[test]
fn printable_decode_filters_control_bytes() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let mut probabilities = [0_i16; BYTE_VOCAB];
    logits[0] = 1000;
    probabilities[0] = 20_000;
    logits[usize::from(b'A')] = 900;
    probabilities[usize::from(b'A')] = 10_000;
    let decode = DecodeConfig {
        printable_only: true,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
        .expect("printable decode");

    assert_eq!(token, b'A');
}

#[test]
fn ascii_lower_decode_filters_outside_curriculum_bytes() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'Z')] = 1000;
    logits[usize::from(b'@')] = 900;
    logits[usize::from(b'z')] = 800;
    let decode = DecodeConfig {
        ascii_lower_only: true,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
        .expect("ascii lower decode");

    assert_eq!(token, b'z');
}

#[test]
fn max_repeat_run_decode_breaks_greedy_loop() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'a')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        max_repeat_run: 3,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"aaa")
        .expect("run-capped decode");

    assert_eq!(token, b'b');
}

#[test]
fn strict_adjacency_decode_rejects_unseen_successors() {
    let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'z')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        strict_adjacency: true,
        ..DecodeConfig::greedy()
    };

    let selection =
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", Some(&priors))
            .expect("strict adjacency decode");

    assert_eq!(selection.token, b'b');
    assert_eq!(selection.candidate_count, 1);
    assert_eq!(selection.rejected_candidates.adjacency, BYTE_VOCAB - 1);
}

#[test]
fn corpus_prior_can_rerank_greedy_decode() {
    let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'z')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        corpus_prior: true,
        corpus_prior_logit_shift: 7,
        ..DecodeConfig::greedy()
    };

    let selection =
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", Some(&priors))
            .expect("corpus prior decode");

    assert_eq!(selection.token, b'b');
    assert_eq!(selection.candidate_count, BYTE_VOCAB);
    assert_eq!(selection.rejected_candidates.adjacency, 0);
}

#[test]
fn corpus_prior_decode_requires_priors() {
    let logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    let decode = DecodeConfig {
        corpus_prior: true,
        ..DecodeConfig::greedy()
    };

    assert!(
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", None).is_err()
    );
}

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

#[test]
fn mini_transformer_swarm_model_roundtrips_and_generates() {
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
    let bytes = run.swarm_model.try_to_bytes().expect("swarm bytes");
    let decoded = MiniTransformerMlpSwarmModel::from_bytes(&bytes).expect("swarm model");

    assert_eq!(decoded, run.swarm_model);
    assert_eq!(decoded.worker_count(), 2);
    assert_eq!(decoded.model_hash(), run.swarm_model.model_hash());
    let manifest = decoded.to_expert_manifest().expect("swarm manifest");

    assert_eq!(manifest.artifact_byte_count, bytes.len());
    assert_eq!(manifest.model_hash, decoded.model_hash());
    assert_eq!(manifest.worker_count, 2);
    assert_eq!(manifest.worker_model_hashes.len(), 2);
    assert_eq!(manifest.worker_parameter_bytes.len(), 2);
    assert_eq!(
        manifest.parameter_bytes,
        manifest.worker_parameter_bytes.iter().sum::<usize>()
    );
    let manifest_json = manifest.to_json_line();
    assert!(
        manifest_json.contains("\"schema\":\"nsrl.mini_transformer_swarm_expert_manifest.v1\"")
    );
    assert!(manifest_json.contains("\"supported_compositions\":[\"average_logits\",\"confidence_weighted\",\"confidence_router\"]"));
    assert!(manifest_json.contains("\"artifact\":{\"format\":\"nsrlswarm\""));
    let mut oversized_manifest = manifest.clone();
    oversized_manifest.parameter_bytes = manifest.parameter_bytes.saturating_add(1);
    let route = route_mini_transformer_swarm_experts(
        &[
            MiniTransformerSwarmRouteCandidate {
                expert_id: String::from("fit.nsrlswarm"),
                manifest: manifest.clone(),
            },
            MiniTransformerSwarmRouteCandidate {
                expert_id: String::from("too-large.nsrlswarm"),
                manifest: oversized_manifest,
            },
        ],
        MiniTransformerSwarmRouteConfig {
            required_capabilities: vec![
                String::from("byte_generation"),
                String::from("integer_q15"),
            ],
            max_artifact_bytes: Some(bytes.len()),
            max_parameter_bytes: Some(manifest.parameter_bytes),
            active_expert_limit: 1,
            prompt_affinity: false,
            prompt_affinity_max_windows: 32,
        },
        b"to be",
    )
    .expect("swarm route");
    assert_eq!(route.selected_expert_indices, vec![0]);
    assert!(route.candidates[0].accepted);
    assert_eq!(
        route.candidates[0].matched_capabilities,
        vec![String::from("byte_generation"), String::from("integer_q15")]
    );
    assert!(route.candidates[0].missing_capabilities.is_empty());
    assert!(!route.candidates[1].accepted);
    assert_eq!(
        route.candidates[1].reject_reason,
        "parameter_budget_exceeded"
    );
    let route_json = route.to_json_line();
    assert!(route_json.contains("\"schema\":\"nsrl.mini_transformer_swarm_route_trace.v1\""));
    assert!(route_json.contains("\"selected_expert_indices\":[0]"));
    let prompt_affinity_route = route_mini_transformer_swarm_expert_models(
        &[
            MiniTransformerSwarmRoutedGenerationExpert {
                expert_id: String::from("left.nsrlswarm"),
                model: decoded.clone(),
            },
            MiniTransformerSwarmRoutedGenerationExpert {
                expert_id: String::from("right.nsrlswarm"),
                model: decoded.clone(),
            },
        ],
        MiniTransformerSwarmRouteConfig {
            required_capabilities: vec![String::from("byte_generation")],
            max_artifact_bytes: Some(bytes.len()),
            max_parameter_bytes: None,
            active_expert_limit: 1,
            prompt_affinity: true,
            prompt_affinity_max_windows: 4,
        },
        b"to be",
        MiniTransformerAttentionKind::Linear,
        MiniTransformerPositionPolicy::Nope,
        MiniTransformerSwarmComposition::ConfidenceRouter,
    )
    .expect("prompt-affinity route");
    assert_eq!(prompt_affinity_route.selected_expert_indices, vec![0]);
    assert!(
        prompt_affinity_route
            .candidates
            .iter()
            .all(|candidate| candidate.prompt_eval_windows > 0
                && candidate.prompt_probability_error_q15.is_some())
    );
    assert!(
        route_mini_transformer_swarm_experts(
            &[MiniTransformerSwarmRouteCandidate {
                expert_id: String::from("fit.nsrlswarm"),
                manifest,
            }],
            MiniTransformerSwarmRouteConfig {
                required_capabilities: vec![String::from("lexeme_generation")],
                max_artifact_bytes: None,
                max_parameter_bytes: None,
                active_expert_limit: 1,
                prompt_affinity: false,
                prompt_affinity_max_windows: 32,
            },
            b"to be",
        )
        .is_err()
    );
    let routed_generation = generate_routed_mini_transformer_swarm_experts(
        &[
            MiniTransformerSwarmRoutedGenerationExpert {
                expert_id: String::from("left.nsrlswarm"),
                model: decoded.clone(),
            },
            MiniTransformerSwarmRoutedGenerationExpert {
                expert_id: String::from("right.nsrlswarm"),
                model: decoded.clone(),
            },
        ],
        MiniTransformerSwarmRouteConfig {
            required_capabilities: vec![String::from("byte_generation")],
            max_artifact_bytes: Some(bytes.len()),
            max_parameter_bytes: None,
            active_expert_limit: 2,
            prompt_affinity: true,
            prompt_affinity_max_windows: 4,
        },
        b"to be",
        ByteGenerationConfig::greedy(2),
        MiniTransformerAttentionKind::Linear,
        MiniTransformerPositionPolicy::Nope,
        MiniTransformerSwarmComposition::ConfidenceRouter,
        None,
    )
    .expect("routed swarm generation");
    assert_eq!(routed_generation.route.selected_expert_indices, vec![0, 1]);
    assert!(routed_generation.route.candidates.iter().all(|candidate| {
        candidate.prompt_eval_windows > 0 && candidate.prompt_probability_error_q15.is_some()
    }));
    assert_eq!(routed_generation.selected_expert_ids.len(), 2);
    assert_eq!(routed_generation.active_worker_count, 4);
    assert_eq!(
        routed_generation.generation.composition,
        MiniTransformerSwarmComposition::ConfidenceRouter
    );
    assert_eq!(routed_generation.generation.generated_bytes.len(), 2);
    assert!(
        routed_generation
            .to_json_line()
            .contains("\"schema\":\"nsrl.mini_transformer_swarm_routed_generation_trace.v1\"")
    );

    let generation =
        generate_mini_transformer_swarm_with_attention_kind_position_policy_and_priors(
            &decoded,
            b"to be",
            ByteGenerationConfig::greedy(4),
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
            None,
        )
        .expect("swarm generation");

    assert_eq!(generation.worker_count, 2);
    assert_eq!(generation.swarm_model_hash, decoded.model_hash());
    assert_eq!(
        generation.composition,
        MiniTransformerSwarmComposition::AverageLogits
    );
    assert_eq!(generation.generated_bytes.len(), 4);
    assert!(
        generation
            .to_json_line()
            .contains("\"schema\":\"nsrl.mini_transformer_swarm_generation_trace.v1\"")
    );

    let weighted_generation =
        generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
            &decoded,
            b"to be",
            ByteGenerationConfig::greedy(2),
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
            MiniTransformerSwarmComposition::ConfidenceWeighted,
            None,
        )
        .expect("weighted swarm generation");
    let router_generation =
        generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
            &decoded,
            b"to be",
            ByteGenerationConfig::greedy(2),
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
            MiniTransformerSwarmComposition::ConfidenceRouter,
            None,
        )
        .expect("router swarm generation");

    assert_eq!(
        weighted_generation.composition,
        MiniTransformerSwarmComposition::ConfidenceWeighted
    );
    assert_eq!(
        router_generation.composition,
        MiniTransformerSwarmComposition::ConfidenceRouter
    );
    assert!(
        router_generation
            .to_json_line()
            .contains("\"composition\":\"confidence_router\"")
    );
}

#[cfg(not(feature = "mini-calibrated"))]
struct MiniTransformerTrainCoreWorkspaceBuffers {
    embedding_output: Vec<i16>,
    attention_norm: Vec<i16>,
    attention_q: Vec<i16>,
    attention_k: Vec<i16>,
    attention_v: Vec<i16>,
    attention_context: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    attention_state_kv: Vec<i64>,
    attention_key_sums: Vec<i64>,
    mlp_norm: Vec<i16>,
    mlp_up: Vec<i16>,
    mlp_gate: Vec<i16>,
    mlp_gated: Vec<i16>,
    mlp_output: Vec<i16>,
    block_output: Vec<i16>,
    logits_q8: Vec<i32>,
    probabilities_q15: Vec<i16>,
    grad_output_q15: Vec<i16>,
    output_scaled_grad: Vec<i32>,
    grad_last_features: Vec<i16>,
    grad_mlp_output: Vec<i16>,
    grad_mlp_input: Vec<i16>,
    mlp_scaled_grad: Vec<i32>,
    mlp_input_grad_gated: Vec<i16>,
    mlp_input_grad_up: Vec<i16>,
    mlp_input_grad_gate: Vec<i16>,
    mlp_input_grad_up_input: Vec<i16>,
    mlp_input_grad_gate_input: Vec<i16>,
    mlp_update_grad_gated: Vec<i16>,
    mlp_update_grad_up: Vec<i16>,
    mlp_update_grad_gate: Vec<i16>,
    grad_attention_output: Vec<i16>,
    grad_attention_context: Vec<i16>,
    attention_scaled_grad: Vec<i32>,
    linear_prefix_states: Vec<i64>,
    linear_denominators: Vec<i64>,
    linear_grad_state_q15: Vec<i64>,
    linear_grad_q_acc: Vec<i64>,
    linear_grad_k_acc: Vec<i64>,
    linear_grad_v_acc: Vec<i64>,
    grad_attention_q: Vec<i16>,
    grad_attention_k: Vec<i16>,
    grad_attention_v: Vec<i16>,
    grad_attention_norm_input: Vec<i16>,
    grad_embedding_output: Vec<i16>,
}

#[cfg(not(feature = "mini-calibrated"))]
impl MiniTransformerTrainCoreWorkspaceBuffers {
    fn new(seq_len: usize) -> Self {
        assert_eq!(
            nsrl_train_core::MINI_TRANSFORMER_D_MODEL,
            MINI_TRANSFORMER_D_MODEL
        );
        assert_eq!(
            nsrl_train_core::MINI_TRANSFORMER_HEADS,
            MINI_TRANSFORMER_HEADS
        );
        assert_eq!(
            nsrl_train_core::MINI_TRANSFORMER_HIDDEN_DIM,
            MINI_TRANSFORMER_HIDDEN_DIM
        );
        assert_eq!(nsrl_train_core::BYTE_VOCAB, BYTE_VOCAB);

        let total = seq_len * MINI_TRANSFORMER_D_MODEL;
        let hidden_total = seq_len * MINI_TRANSFORMER_HIDDEN_DIM;
        let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
        let head_state_len = head_dim * head_dim;
        let state_len = MINI_TRANSFORMER_HEADS * head_state_len;
        let key_sum_len = MINI_TRANSFORMER_HEADS * head_dim;
        let prefix_len = seq_len * state_len;
        let denom_len = seq_len * MINI_TRANSFORMER_HEADS;
        let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

        Self {
            embedding_output: vec![0_i16; total],
            attention_norm: vec![0_i16; total],
            attention_q: vec![0_i16; total],
            attention_k: vec![0_i16; total],
            attention_v: vec![0_i16; total],
            attention_context: vec![0_i16; total],
            attention_output: vec![0_i16; total],
            attention_residual: vec![0_i16; total],
            attention_state_kv: vec![0_i64; state_len],
            attention_key_sums: vec![0_i64; key_sum_len],
            mlp_norm: vec![0_i16; total],
            mlp_up: vec![0_i16; hidden_total],
            mlp_gate: vec![0_i16; hidden_total],
            mlp_gated: vec![0_i16; hidden_total],
            mlp_output: vec![0_i16; total],
            block_output: vec![0_i16; total],
            logits_q8: vec![0_i32; BYTE_VOCAB],
            probabilities_q15: vec![0_i16; BYTE_VOCAB],
            grad_output_q15: vec![0_i16; BYTE_VOCAB],
            output_scaled_grad: vec![0_i32; BYTE_VOCAB],
            grad_last_features: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
            grad_mlp_output: vec![0_i16; total],
            grad_mlp_input: vec![0_i16; total],
            mlp_scaled_grad: vec![0_i32; scaled_len],
            mlp_input_grad_gated: vec![0_i16; hidden_total],
            mlp_input_grad_up: vec![0_i16; hidden_total],
            mlp_input_grad_gate: vec![0_i16; hidden_total],
            mlp_input_grad_up_input: vec![0_i16; total],
            mlp_input_grad_gate_input: vec![0_i16; total],
            mlp_update_grad_gated: vec![0_i16; hidden_total],
            mlp_update_grad_up: vec![0_i16; hidden_total],
            mlp_update_grad_gate: vec![0_i16; hidden_total],
            grad_attention_output: vec![0_i16; total],
            grad_attention_context: vec![0_i16; total],
            attention_scaled_grad: vec![0_i32; MINI_TRANSFORMER_D_MODEL],
            linear_prefix_states: vec![0_i64; prefix_len],
            linear_denominators: vec![0_i64; denom_len],
            linear_grad_state_q15: vec![0_i64; head_state_len],
            linear_grad_q_acc: vec![0_i64; total],
            linear_grad_k_acc: vec![0_i64; total],
            linear_grad_v_acc: vec![0_i64; total],
            grad_attention_q: vec![0_i16; total],
            grad_attention_k: vec![0_i16; total],
            grad_attention_v: vec![0_i16; total],
            grad_attention_norm_input: vec![0_i16; total],
            grad_embedding_output: vec![0_i16; total],
        }
    }

    fn as_workspace(&mut self) -> nsrl_train_core::MiniTransformerStepWorkspace<'_> {
        nsrl_train_core::MiniTransformerStepWorkspace {
            embedding_output: &mut self.embedding_output,
            attention_norm: &mut self.attention_norm,
            attention_q: &mut self.attention_q,
            attention_k: &mut self.attention_k,
            attention_v: &mut self.attention_v,
            attention_context: &mut self.attention_context,
            attention_output: &mut self.attention_output,
            attention_residual: &mut self.attention_residual,
            attention_state_kv: &mut self.attention_state_kv,
            attention_key_sums: &mut self.attention_key_sums,
            mlp_norm: &mut self.mlp_norm,
            mlp_up: &mut self.mlp_up,
            mlp_gate: &mut self.mlp_gate,
            mlp_gated: &mut self.mlp_gated,
            mlp_output: &mut self.mlp_output,
            block_output: &mut self.block_output,
            logits_q8: &mut self.logits_q8,
            probabilities_q15: &mut self.probabilities_q15,
            grad_output_q15: &mut self.grad_output_q15,
            output_scaled_grad: &mut self.output_scaled_grad,
            grad_last_features: &mut self.grad_last_features,
            grad_mlp_output: &mut self.grad_mlp_output,
            grad_mlp_input: &mut self.grad_mlp_input,
            mlp_scaled_grad: &mut self.mlp_scaled_grad,
            mlp_input_grad_gated: &mut self.mlp_input_grad_gated,
            mlp_input_grad_up: &mut self.mlp_input_grad_up,
            mlp_input_grad_gate: &mut self.mlp_input_grad_gate,
            mlp_input_grad_up_input: &mut self.mlp_input_grad_up_input,
            mlp_input_grad_gate_input: &mut self.mlp_input_grad_gate_input,
            mlp_update_grad_gated: &mut self.mlp_update_grad_gated,
            mlp_update_grad_up: &mut self.mlp_update_grad_up,
            mlp_update_grad_gate: &mut self.mlp_update_grad_gate,
            grad_attention_output: &mut self.grad_attention_output,
            grad_attention_context: &mut self.grad_attention_context,
            attention_scaled_grad: &mut self.attention_scaled_grad,
            linear_prefix_states: &mut self.linear_prefix_states,
            linear_denominators: &mut self.linear_denominators,
            linear_grad_state_q15: &mut self.linear_grad_state_q15,
            linear_grad_q_acc: &mut self.linear_grad_q_acc,
            linear_grad_k_acc: &mut self.linear_grad_k_acc,
            linear_grad_v_acc: &mut self.linear_grad_v_acc,
            grad_attention_q: &mut self.grad_attention_q,
            grad_attention_k: &mut self.grad_attention_k,
            grad_attention_v: &mut self.grad_attention_v,
            grad_attention_norm_input: &mut self.grad_attention_norm_input,
            grad_embedding_output: &mut self.grad_embedding_output,
        }
    }
}

#[cfg(not(feature = "mini-calibrated"))]
#[test]
fn mini_transformer_train_core_linear_nope_step_matches_std_single_window() {
    let tokens = b"To be";
    let seq_len = 4;
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len,
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
        attention_kind: MiniTransformerAttentionKind::Linear,
        position_policy: MiniTransformerPositionPolicy::Nope,
        learning_rate: 1,
        output_learning_rate_shift: 18,
        mlp_learning_rate_shift: 17,
        embedding_learning_rate_shift: 13,
        attention_learning_rate_shift: 22,
        attention_q_learning_rate_shift: 18,
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
    };
    let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(seq_len, 1)
        .expect("single-layer model");
    let std_run =
        run_mini_transformer_mlp_training_from_model(tokens, config, initial_model.clone())
            .expect("std training");
    assert_eq!(std_run.trace.updates, 1);
    assert_eq!(std_run.trace.rollback_count, 0);
    assert_eq!(std_run.trace.rejected_window_count, 0);

    let mut core_model = initial_model;
    let core_stats = {
        let mut model_slices = nsrl_train_core::MiniTransformerModelSlicesMut {
            embeddings: &mut core_model.embeddings,
            q_weights: &mut core_model.q_weights,
            k_weights: &mut core_model.k_weights,
            v_weights: &mut core_model.v_weights,
            o_weights: &mut core_model.o_weights,
            up_weights: &mut core_model.up_weights,
            gate_weights: &mut core_model.gate_weights,
            down_weights: &mut core_model.down_weights,
            output_weights: &mut core_model.output_weights,
        };
        let mut buffers = MiniTransformerTrainCoreWorkspaceBuffers::new(seq_len);
        let mut workspace = buffers.as_workspace();
        nsrl_train_core::mini_transformer_linear_nope_train_step(
            &mut model_slices,
            &tokens[..seq_len],
            tokens[seq_len],
            nsrl_train_core::MiniTransformerStepConfig {
                seq_len,
                learning_rate: config.learning_rate,
                output_learning_rate_shift: config.output_learning_rate_shift,
                mlp_learning_rate_shift: config.mlp_learning_rate_shift,
                embedding_learning_rate_shift: config.embedding_learning_rate_shift,
                attention_learning_rate_shift: config.attention_learning_rate_shift,
                attention_q_learning_rate_shift: config.attention_q_learning_rate_shift,
                attention_qk_learning_rate_shift: config.attention_qk_learning_rate_shift,
            },
            &mut workspace,
        )
        .expect("train core step")
    };
    let std_step = &std_run.trace.steps[0];
    assert_eq!(core_stats.predicted_before, std_step.predicted_token_before);
    assert_eq!(core_stats.predicted_after, std_step.predicted_token_after);
    assert_eq!(
        core_stats.output_head.gradient_saturation_count,
        std_step.output_head_saturation_count
    );
    assert_eq!(
        core_stats.output_head.zero_delta_count,
        std_step.output_head_zero_delta_count
    );
    assert_eq!(
        core_stats.output_head.weight_delta_l1,
        std_step.output_head_delta_l1
    );
    assert_eq!(
        core_stats.mlp.gradient_saturation_count(),
        std_step.mlp_saturation_count
    );
    assert_eq!(
        core_stats.mlp.zero_delta_count(),
        std_step.mlp_zero_delta_count
    );
    assert_eq!(core_stats.mlp.weight_delta_l1(), std_step.mlp_delta_l1);
    assert_eq!(
        core_stats.embedding.gradient_saturation_count,
        std_step.embedding_saturation_count
    );
    assert_eq!(
        core_stats.embedding.zero_delta_count,
        std_step.embedding_zero_delta_count
    );
    assert_eq!(
        core_stats.embedding.weight_delta_l1,
        std_step.embedding_delta_l1
    );
    assert_eq!(
        core_stats.attention.gradient_saturation_count(),
        std_step.attention_saturation_count
    );
    assert_eq!(
        core_stats.attention.zero_delta_count(),
        std_step.attention_zero_delta_count
    );
    assert_eq!(
        core_stats.attention.weight_delta_l1(),
        std_step.attention_delta_l1
    );
    assert_eq!(
        core_stats.attention.q.weight_delta_l1,
        std_step.attention_q_delta_l1
    );
    assert_eq!(
        core_stats.attention.k.weight_delta_l1,
        std_step.attention_k_delta_l1
    );
    assert_eq!(
        core_stats.attention.v.weight_delta_l1,
        std_step.attention_v_delta_l1
    );
    assert_eq!(
        core_stats.attention.o.weight_delta_l1,
        std_step.attention_o_delta_l1
    );
    assert_eq!(
        core_stats.residual_saturation_count,
        std_step.residual_saturation_count
    );
    assert_eq!(core_model.embeddings, std_run.model.embeddings);
    assert_eq!(
        core_model.position_embeddings,
        std_run.model.position_embeddings
    );
    assert_eq!(core_model.q_weights, std_run.model.q_weights);
    assert_eq!(core_model.k_weights, std_run.model.k_weights);
    assert_eq!(core_model.v_weights, std_run.model.v_weights);
    assert_eq!(core_model.o_weights, std_run.model.o_weights);
    assert_eq!(core_model.up_weights, std_run.model.up_weights);
    assert_eq!(core_model.gate_weights, std_run.model.gate_weights);
    assert_eq!(core_model.down_weights, std_run.model.down_weights);
    assert_eq!(core_model.output_weights, std_run.model.output_weights);
}

#[test]
fn mini_transformer_nope_training_does_not_update_position_embeddings() {
    let tokens = b"To be or not to be, that is the question. To be or not to be. ";
    let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
        .expect("single-layer model");
    let initial_positions = initial_model.position_embeddings.clone();
    let initial_token_embedding_hash = hash_i16_slice(&initial_model.embeddings);
    let run = run_mini_transformer_mlp_training_from_model(
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
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
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
        },
        initial_model,
    )
    .expect("nope train");

    assert_eq!(run.model.position_embeddings, initial_positions);
    assert_ne!(
        hash_i16_slice(&run.model.embeddings),
        initial_token_embedding_hash
    );
    assert!(run.trace.to_json_line().contains("\"position\":\"nope\""));
}

#[test]
fn mini_transformer_batch_windows_are_traced() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
        .expect("single-layer model");
    let trace = run_mini_transformer_mlp_training_from_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(8),
            batch_windows: 4,
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
        initial_model,
    )
    .expect("mini batch train")
    .trace;

    assert_eq!(trace.examined_windows, 8);
    assert_eq!(trace.accepted_batch_count + trace.rejected_batch_count, 2);
    assert_eq!(trace.mlp_accumulator_window_count, trace.updates);
    assert_eq!(trace.attention_accumulator_window_count, trace.updates);
    assert_eq!(trace.embedding_accumulator_window_count, trace.updates);
    let line = trace.to_json_line();
    assert!(line.contains("\"batch_windows\":4"));
    assert!(line.contains("\"batch_mode\":\"serial\""));
    assert!(line.contains("\"map_reduce_workers\":1"));
    assert!(line.contains("\"batch_average_shift\":2"));
    assert!(line.contains("\"mlp_accumulator_window_count\""));
    assert!(line.contains("\"attention_accumulator_window_count\""));
    assert!(line.contains("\"embedding_accumulator_window_count\""));
}

#[test]
fn mini_transformer_map_reduce_single_worker_smoke_is_traced() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let trace = run_mini_transformer_mlp_training(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(4),
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
            mlp_learning_rate_shift: 17,
            embedding_learning_rate_shift: 13,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 16,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 1,
        },
    )
    .expect("map-reduce smoke");

    assert_eq!(trace.examined_windows, 4);
    assert_eq!(trace.accepted_batch_count + trace.rejected_batch_count, 2);
    let line = trace.to_json_line();
    assert!(line.contains("\"batch_mode\":\"map-reduce\""));
    assert!(line.contains("\"map_reduce_workers\":1"));
    assert!(line.contains("\"effective_map_reduce_workers\":1"));
}

#[test]
fn mini_transformer_map_reduce_stacked_accumulates_lower_layers_and_embeddings() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
    assert_eq!(initial_model.transformer_layers(), 2);
    let config = MiniTransformerMlpTrainConfig {
        epochs: 1,
        seq_len: 4,
        stride: 1,
        window_offset: 0,
        max_windows: Some(4),
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
        mlp_learning_rate_shift: 17,
        embedding_learning_rate_shift: 13,
        attention_learning_rate_shift: 22,
        attention_q_learning_rate_shift: 18,
        attention_qk_learning_rate_shift: 16,
        adaptive_rule_shifts: false,
        adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
        adaptive_attention_shifts: false,
        adaptive_holographic_shifts: false,
        attention_vo_error_feedback: false,
        attention_vo_oracle: false,
        reject_loss_regression: false,
        batch_mode: MiniTransformerBatchMode::MapReduce,
        map_reduce_workers: 1,
    };
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
        tokens,
        &starts,
        config.seq_len,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )
    .expect("frequency weights");
    let batch_result = mini_transformer_map_reduce_batch(
        tokens,
        &starts,
        &target_frequency_weights_q15,
        0,
        config.batch_windows,
        0,
        &initial_model,
        config,
        0,
        MiniTransformerTraceDetail::None,
        1,
    )
    .expect("stacked map-reduce batch");

    assert_eq!(batch_result.mlp_weight_gradients.len(), 2);
    assert_eq!(batch_result.attention_weight_gradients.len(), 2);
    assert!(batch_result.embedding_gradient.sample_count > 0);
    assert!(
        batch_result.mlp_weight_gradients[0]
            .down
            .accumulators
            .iter()
            .any(|&value| value != 0)
    );
    assert!(
        batch_result.attention_weight_gradients[0]
            .o
            .accumulators
            .iter()
            .any(|&value| value != 0)
    );

    let run = run_mini_transformer_mlp_training_from_model(tokens, config, initial_model)
        .expect("stacked map-reduce train");
    assert_eq!(run.model.transformer_layers(), 2);
    assert_eq!(run.trace.mlp_accumulator_window_count, run.trace.updates);
    assert_eq!(
        run.trace.attention_accumulator_window_count,
        run.trace.updates
    );
    assert_eq!(
        run.trace.embedding_accumulator_window_count,
        run.trace.updates
    );
}

#[test]
fn mini_transformer_map_reduce_multi_worker_matches_single_worker() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let single_worker = run_mini_transformer_mlp_training_with_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            epochs: 1,
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(6),
            batch_windows: 3,
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
            mlp_learning_rate_shift: 17,
            embedding_learning_rate_shift: 13,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 16,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 1,
        },
    )
    .expect("single-worker map-reduce");
    let multi_worker = run_mini_transformer_mlp_training_with_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            map_reduce_workers: 3,
            ..single_worker.trace.config
        },
    )
    .expect("multi-worker map-reduce");

    assert_eq!(multi_worker.trace.examined_windows, 6);
    assert_eq!(
        multi_worker.trace.accepted_batch_count + multi_worker.trace.rejected_batch_count,
        2
    );
    assert_eq!(multi_worker.model, single_worker.model);
    assert_eq!(
        multi_worker.trace.final_model_hash,
        single_worker.trace.final_model_hash
    );
    assert_eq!(
        multi_worker.trace.output_head_accumulator_window_count,
        single_worker.trace.output_head_accumulator_window_count
    );
    assert_eq!(
        multi_worker.trace.attention_accumulator_window_count,
        single_worker.trace.attention_accumulator_window_count
    );
    let line = multi_worker.trace.to_json_line();
    assert!(line.contains("\"batch_mode\":\"map-reduce\""));
    assert!(line.contains("\"map_reduce_workers\":3"));
    assert!(line.contains("\"effective_map_reduce_workers\":3"));
}

#[test]
fn mini_transformer_map_reduce_matches_serial_with_ascii_lower_adaptive_rule_shifts() {
    let tokens =
        b"To be or not to be, that is the question. To be or not to be, that is the question. ";
    let initial_model = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(4, 1)
        .expect("single-layer model");
    let serial = run_mini_transformer_mlp_training_from_model(
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
            tokenizer_id: ByteTokenizerId::AsciiLowerText,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 17,
            embedding_learning_rate_shift: 13,
            attention_learning_rate_shift: 22,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 16,
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
        initial_model.clone(),
    )
    .expect("serial adaptive ascii-lower");
    let single_worker = run_mini_transformer_mlp_training_from_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            batch_mode: MiniTransformerBatchMode::MapReduce,
            ..serial.trace.config
        },
        initial_model.clone(),
    )
    .expect("single-worker map-reduce adaptive ascii-lower");
    let multi_worker = run_mini_transformer_mlp_training_from_model(
        tokens,
        MiniTransformerMlpTrainConfig {
            batch_mode: MiniTransformerBatchMode::MapReduce,
            map_reduce_workers: 3,
            ..serial.trace.config
        },
        initial_model,
    )
    .expect("multi-worker map-reduce adaptive ascii-lower");

    assert_eq!(serial.model, single_worker.model);
    assert_eq!(serial.model, multi_worker.model);
    assert_eq!(
        serial.trace.final_model_hash,
        single_worker.trace.final_model_hash
    );
    assert_eq!(
        serial.trace.final_model_hash,
        multi_worker.trace.final_model_hash
    );
    assert!(multi_worker.trace.adaptive_rule_update_count > 0);
    let line = multi_worker.trace.to_json_line();
    assert!(line.contains("\"tokenizer\":\"byte_ascii_lower_text_u8_v1\""));
    assert!(line.contains("\"adaptive_rule_shifts\":true"));
    assert!(line.contains("\"batch_mode\":\"map-reduce\""));
}

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

#[test]
fn byte_window_starts_remain_sequential() {
    let starts = byte_window_starts(1000, 4, 10, 0, Some(5));
    assert_eq!(starts, vec![0, 10, 20, 30, 40]);
}

#[test]
fn mini_transformer_window_starts_spread_capped_runs() {
    let starts = mini_transformer_window_starts(1000, 4, 10, 0, Some(5));
    assert_eq!(starts, vec![0, 250, 500, 740, 990]);
}

#[test]
fn mini_transformer_window_starts_keep_full_runs_sequential() {
    let sequential = byte_window_starts(1000, 4, 10, 0, None);
    let distributed = mini_transformer_window_starts(1000, 4, 10, 0, None);
    assert_eq!(distributed, sequential);
}

#[test]
fn mini_transformer_filtered_window_starts_cap_after_target_filter() {
    let mut tokens = vec![b'a'; 40];
    for target_index in [4_usize, 10, 16, 22, 28, 34] {
        tokens[target_index] = b'Z';
    }

    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(3),
            target_token_min: b'Z',
            target_token_max: b'Z',
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![0, 18, 30]);
    assert!(starts.iter().all(|&start| tokens[start + 4] == b'Z'));
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_marker_segment() {
    let tokens = [
        0, 1, 2, b's', b'e', 3, b'A', b'B', 5, 1, 2, b'x', 3, b'C', 4, b'i', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'A',
            target_token_max: b'Z',
            target_segment: MiniTransformerTargetSegment::after_marker_before_any(3, &[4, 5])
                .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![5, 6, 12]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'A', b'B', b'C']
    );
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_sequence_segment() {
    let tokens = [
        1, 2, b'p', 3, b'S', b'o', b'B', b'a', b':', b' ', b'H', 5, 1, 2, b'q', 3, b'S', b'o',
        b'C', b'a', b'm', b':', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'A',
            target_token_max: b'z',
            target_segment: MiniTransformerTargetSegment::after_sequence_before_any(
                &[3, b'S', b'o'],
                &[b':', 4, 5],
            )
            .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![5, 6, 17, 18, 19]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'B', b'a', b'C', b'a', b'm']
    );
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_first_after_sequence() {
    let tokens = [
        1, 3, b'H', b'e', b' ', b'm', b'a', 5, 1, 3, b'H', b'e', b' ', b'i', b's', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'a',
            target_token_max: b'z',
            target_segment: MiniTransformerTargetSegment::first_after_sequence_before_any(
                b"He ",
                &[4, 5],
            )
            .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![4, 12]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'm', b'i']
    );
}

#[test]
fn mini_transformer_loss_guard_starts_mix_batch_and_global_points() {
    let starts: Vec<usize> = (0..32).map(|index| index * 10).collect();
    let guarded = mini_transformer_loss_guard_starts(&starts, 5, 7);

    assert!(guarded.contains(&50));
    assert!(guarded.contains(&60));
    assert_eq!(guarded.first().copied(), Some(50));
    assert_eq!(guarded.get(1).copied(), Some(60));
    assert!(guarded.contains(&0));
    assert!(guarded.contains(&310));
    assert_eq!(guarded.len(), 17);
}

#[test]
fn mini_transformer_loss_guard_ignores_small_regressions() {
    assert!(!mini_transformer_loss_guard_regressed(100_000, 117_000, 17));
    assert!(mini_transformer_loss_guard_regressed(100_000, 118_000, 17));
}

#[test]
fn attention_vo_oracle_does_not_increase_configured_loss() {
    let tokens = b"To be or not to be, that is the question. To be or not to be. ";
    let seq_len = 4;
    let starts = byte_window_starts(tokens.len(), seq_len, 1, 0, Some(4));
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(seq_len);
    if MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL {
        assert_eq!(
            mini_transformer_attention_vo_oracle_update_i8_checked(
                &mut model, tokens, &starts, seq_len, 1,
            ),
            Err(TrainError::InvalidConfig)
        );
        return;
    }
    let before = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
        .expect("before loss");
    let (v, o) = mini_transformer_attention_vo_oracle_update_i8_checked(
        &mut model, tokens, &starts, seq_len, 1,
    )
    .expect("oracle update");
    let after = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
        .expect("after loss");

    assert!(after <= before);
    assert_eq!(v.gradient_saturation_count, 0);
    assert_eq!(o.gradient_saturation_count, 0);
    assert_eq!(
        v.zero_delta_count + v.weight_delta_l1 as usize,
        MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
    );
    assert_eq!(
        o.zero_delta_count + o.weight_delta_l1 as usize,
        MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
    );
}

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
