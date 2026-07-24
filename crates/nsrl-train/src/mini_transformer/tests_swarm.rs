//! Mini-transformer tests — swarm.
use super::*;

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
