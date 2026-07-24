//! Mini-transformer tests — ser.
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
