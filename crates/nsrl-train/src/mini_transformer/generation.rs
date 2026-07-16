//! Mini-transformer routing and generation entry points.

use super::*;

pub fn route_mini_transformer_swarm_experts(
    candidates: &[MiniTransformerSwarmRouteCandidate],
    config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    route_mini_transformer_swarm_experts_with_prompt_affinity(candidates, config, prompt, None)
}

pub fn route_mini_transformer_swarm_expert_models(
    experts: &[MiniTransformerSwarmRoutedGenerationExpert],
    route_config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    let candidates = experts
        .iter()
        .map(|expert| {
            Ok(MiniTransformerSwarmRouteCandidate {
                expert_id: expert.expert_id.clone(),
                manifest: expert.model.to_expert_manifest()?,
            })
        })
        .collect::<Result<Vec<_>, TrainError>>()?;
    let prompt_affinities = if route_config.prompt_affinity {
        Some(
            experts
                .iter()
                .map(|expert| {
                    mini_transformer_swarm_prompt_affinity(
                        &expert.model,
                        prompt,
                        attention_kind,
                        position_policy,
                        composition,
                        route_config.prompt_affinity_max_windows,
                    )
                })
                .collect::<Result<Vec<_>, TrainError>>()?,
        )
    } else {
        None
    };
    route_mini_transformer_swarm_experts_with_prompt_affinity(
        &candidates,
        route_config,
        prompt,
        prompt_affinities.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_routed_mini_transformer_swarm_experts(
    experts: &[MiniTransformerSwarmRoutedGenerationExpert],
    route_config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmRoutedGenerationTrace, TrainError> {
    let route = route_mini_transformer_swarm_expert_models(
        experts,
        route_config,
        prompt,
        attention_kind,
        position_policy,
        composition,
    )?;
    let mut selected_expert_ids = Vec::with_capacity(route.selected_expert_indices.len());
    let mut active_workers = Vec::new();
    let mut best_worker_index = None;

    for &expert_index in &route.selected_expert_indices {
        let expert = experts.get(expert_index).ok_or(TrainError::InvalidConfig)?;
        selected_expert_ids.push(expert.expert_id.clone());
        let worker_offset = active_workers.len();
        if best_worker_index.is_none() {
            best_worker_index = Some(worker_offset.saturating_add(expert.model.best_worker_index));
        }
        active_workers.extend(expert.model.workers.iter().cloned());
    }

    let active_model =
        MiniTransformerMlpSwarmModel::new(best_worker_index.unwrap_or(0), active_workers)?;
    let generation =
        generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
            &active_model,
            prompt,
            config,
            attention_kind,
            position_policy,
            composition,
            decode_priors,
        )?;

    Ok(MiniTransformerSwarmRoutedGenerationTrace {
        route,
        selected_expert_ids,
        active_worker_count: active_model.worker_count(),
        generation,
    })
}

fn route_mini_transformer_swarm_experts_with_prompt_affinity(
    candidates: &[MiniTransformerSwarmRouteCandidate],
    config: MiniTransformerSwarmRouteConfig,
    prompt: &[u8],
    prompt_affinities: Option<&[MiniTransformerSwarmPromptAffinityTrace]>,
) -> Result<MiniTransformerSwarmRouteDecisionTrace, TrainError> {
    if candidates.is_empty() || config.active_expert_limit == 0 {
        return Err(TrainError::InvalidConfig);
    }
    if let Some(affinities) = prompt_affinities
        && affinities.len() != candidates.len()
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut candidate_traces = Vec::with_capacity(candidates.len());
    for (expert_index, candidate) in candidates.iter().enumerate() {
        candidate_traces.push(mini_transformer_swarm_route_candidate_trace(
            expert_index,
            candidate,
            &config,
            prompt_affinities.and_then(|affinities| affinities.get(expert_index)),
        ));
    }

    let mut selected = candidate_traces
        .iter()
        .filter(|candidate| candidate.accepted)
        .collect::<Vec<_>>();
    selected.sort_by_key(|candidate| {
        (
            core::cmp::Reverse(candidate.score),
            candidate.parameter_bytes,
            candidate.artifact_bytes,
            candidate.expert_index,
        )
    });
    let selected_expert_indices = selected
        .into_iter()
        .take(config.active_expert_limit)
        .map(|candidate| candidate.expert_index)
        .collect::<Vec<_>>();
    if selected_expert_indices.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    Ok(MiniTransformerSwarmRouteDecisionTrace {
        config,
        prompt_bytes: prompt.to_vec(),
        selected_expert_indices,
        candidates: candidate_traces,
    })
}

fn mini_transformer_swarm_route_candidate_trace(
    expert_index: usize,
    candidate: &MiniTransformerSwarmRouteCandidate,
    config: &MiniTransformerSwarmRouteConfig,
    prompt_affinity: Option<&MiniTransformerSwarmPromptAffinityTrace>,
) -> MiniTransformerSwarmRouteCandidateTrace {
    let (capability_match, matched_capabilities, missing_capabilities) =
        mini_transformer_swarm_route_capability_match(
            &candidate.manifest,
            &config.required_capabilities,
        );
    let reject_reason = if !capability_match {
        "capability_mismatch"
    } else if config
        .max_artifact_bytes
        .is_some_and(|max| candidate.manifest.artifact_byte_count > max)
    {
        "artifact_budget_exceeded"
    } else if config
        .max_parameter_bytes
        .is_some_and(|max| candidate.manifest.parameter_bytes > max)
    {
        "parameter_budget_exceeded"
    } else {
        ""
    };
    let accepted = reject_reason.is_empty();
    let manifest_score = if accepted {
        mini_transformer_swarm_route_score(&candidate.manifest, capability_match)
    } else {
        0
    };
    let prompt_affinity_score = if accepted {
        prompt_affinity.map(|affinity| affinity.score).unwrap_or(0)
    } else {
        0
    };
    let score = manifest_score.saturating_add(prompt_affinity_score);

    MiniTransformerSwarmRouteCandidateTrace {
        expert_index,
        expert_id: candidate.expert_id.clone(),
        accepted,
        reject_reason,
        score,
        manifest_score,
        prompt_affinity_score,
        prompt_eval_windows: prompt_affinity
            .map(|affinity| affinity.eval_windows)
            .unwrap_or(0),
        prompt_probability_error_q15: prompt_affinity
            .map(|affinity| affinity.probability_error_q15),
        capability_match,
        matched_capabilities,
        missing_capabilities,
        model_hash: candidate.manifest.model_hash,
        artifact_bytes: candidate.manifest.artifact_byte_count,
        parameter_bytes: candidate.manifest.parameter_bytes,
        worker_count: candidate.manifest.worker_count,
        context_seq_len: candidate.manifest.context_seq_len,
        default_composition: "average_logits",
    }
}

fn mini_transformer_swarm_route_capability_match(
    manifest: &MiniTransformerMlpSwarmExpertManifest,
    required_capabilities: &[String],
) -> (bool, Vec<String>, Vec<String>) {
    let mut matched = Vec::new();
    let mut missing = Vec::new();
    for capability in required_capabilities {
        if matched.iter().any(|seen| seen == capability)
            || missing.iter().any(|seen| seen == capability)
        {
            continue;
        }
        if manifest.supports_capability(capability) {
            matched.push(capability.clone());
        } else {
            missing.push(capability.clone());
        }
    }
    (missing.is_empty(), matched, missing)
}

fn mini_transformer_swarm_route_score(
    manifest: &MiniTransformerMlpSwarmExpertManifest,
    capability_match: bool,
) -> i64 {
    let capability_score = if capability_match { 1_000_000_i64 } else { 0 };
    let worker_score = i64::try_from(manifest.worker_count.min(4096)).unwrap_or(i64::MAX) * 1_000;
    let context_score = i64::try_from(manifest.context_seq_len.min(4096)).unwrap_or(i64::MAX);
    let budget_penalty =
        i64::try_from((manifest.parameter_bytes / 4096).min(i64::MAX as usize)).unwrap_or(i64::MAX);
    capability_score
        .saturating_add(worker_score)
        .saturating_add(context_score)
        .saturating_sub(budget_penalty)
}

impl MiniTransformerSwarmRouteDecisionTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_ROUTE_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_mini_transformer_swarm_route_config_field(&mut out, "config", &self.config);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_hash_field(&mut out, "hash", hash_u8_slice(&self.prompt_bytes));
        out.push('}');
        comma(&mut out);
        push_usize_array_field(
            &mut out,
            "selected_expert_indices",
            &self.selected_expert_indices,
        );
        comma(&mut out);
        push_mini_transformer_swarm_route_candidates_field(
            &mut out,
            "candidates",
            &self.candidates,
        );
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "deterministic_router_not_trained_router_weights",
                "prompt_affinity_is_fixed_prompt_replay_when_enabled",
                "does_not_run_generation",
                "does_not_measure_request_latency_yet",
                "does_not_rank_semantic_quality",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerSwarmRoutedGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            MINI_TRANSFORMER_SWARM_ROUTED_GENERATION_SCHEMA,
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "selected_expert_ids",
            &self
                .selected_expert_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        comma(&mut out);
        push_usize_field(&mut out, "active_worker_count", self.active_worker_count);
        comma(&mut out);
        push_json_line_object_field(&mut out, "route", &self.route.to_json_line());
        comma(&mut out);
        push_json_line_object_field(&mut out, "generation", &self.generation.to_json_line());
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "routes_by_manifest_and_optional_prompt_affinity_not_trained_semantic_router",
                "active_set_workers_are_concatenated_before_generation",
                "does_not_train_router_weights_yet",
                "does_not_claim_language_model_quality",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", MINI_TRANSFORMER_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_string_field(&mut out, "attention_kind", self.attention_kind.as_str());
        comma(&mut out);
        push_string_field(&mut out, "position_policy", self.position_policy.as_str());
        comma(&mut out);
        push_bool_field(
            &mut out,
            "incremental_attention_state",
            self.attention_kind.uses_incremental_state(),
        );
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.prompt_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"generation\":{");
        push_usize_field(&mut out, "new_tokens", self.generated_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.generated_bytes);
        out.push('}');
        comma(&mut out);
        push_mini_transformer_ttt_stats_field(&mut out, "ttt", self.ttt_stats);
        comma(&mut out);
        push_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        let known_non_claims: &[&str] = if self.attention_kind.uses_incremental_state() {
            &MINI_TRANSFORMER_STREAMING_GENERATION_KNOWN_NON_CLAIMS
        } else if self.position_policy == MiniTransformerPositionPolicy::Nope {
            &MINI_TRANSFORMER_NOPE_GENERATION_KNOWN_NON_CLAIMS
        } else {
            &MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS
        };
        push_string_array_field(&mut out, "known_non_claims", known_non_claims);
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerSwarmGenerationTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", GENERATION_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", MINI_TRANSFORMER_SWARM_MODEL_ID);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_string_field(&mut out, "attention_kind", self.attention_kind.as_str());
        comma(&mut out);
        push_string_field(&mut out, "position_policy", self.position_policy.as_str());
        comma(&mut out);
        push_string_field(&mut out, "composition", self.composition.as_str());
        comma(&mut out);
        push_bool_field(
            &mut out,
            "incremental_attention_state",
            self.attention_kind.uses_incremental_state(),
        );
        comma(&mut out);
        push_decode_config_field(&mut out, "decode", self.config);
        comma(&mut out);
        push_decode_priors_field(&mut out, "decode_priors", self.decode_priors);
        comma(&mut out);
        push_hash_field(&mut out, "swarm_model_hash", self.swarm_model_hash);
        comma(&mut out);
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        out.push_str("\"prompt\":{");
        push_usize_field(&mut out, "bytes", self.prompt_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.prompt_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"generation\":{");
        push_usize_field(&mut out, "new_tokens", self.generated_bytes.len());
        comma(&mut out);
        push_u8_array_field(&mut out, "tokens", &self.generated_bytes);
        out.push('}');
        comma(&mut out);
        push_generation_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        let known_non_claims: &[&str] =
            if self.position_policy == MiniTransformerPositionPolicy::Nope {
                &MINI_TRANSFORMER_NOPE_GENERATION_KNOWN_NON_CLAIMS
            } else {
                &MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS
            };
        push_string_array_field(&mut out, "known_non_claims", known_non_claims);
        out.push('}');
        out.push('\n');
        out
    }
}

pub fn generate_mini_transformer(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_priors(model, prompt, config, None)
}

pub fn generate_mini_transformer_swarm(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
        model,
        prompt,
        config,
        MiniTransformerAttentionKind::Linear,
        MiniTransformerPositionPolicy::Nope,
        MiniTransformerSwarmComposition::AverageLogits,
        None,
    )
}

pub fn generate_mini_transformer_swarm_with_attention_kind_position_policy_and_priors(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        position_policy,
        MiniTransformerSwarmComposition::AverageLogits,
        decode_priors,
    )
}

pub fn generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerSwarmGenerationTrace, TrainError> {
    if prompt.is_empty()
        || model.context_seq_len == 0
        || model.workers.is_empty()
        || attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let row = mini_transformer_swarm_ensemble_row_for_context(
            model,
            context_window,
            attention_kind,
            position_policy,
            composition,
        )?;
        let selection = select_byte_from_row_with_priors(
            &row.logits_q8,
            &row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: row.logits_q8[predicted_index],
            predicted_probability_q15: row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });
    }

    Ok(MiniTransformerSwarmGenerationTrace {
        config,
        attention_kind,
        position_policy,
        composition,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        swarm_model_hash: model.model_hash(),
        worker_count: model.worker_count(),
        best_worker_index: model.best_worker_index,
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        steps,
    })
}

fn mini_transformer_swarm_ensemble_row_for_context(
    model: &MiniTransformerMlpSwarmModel,
    context_window: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
) -> Result<ByteVocabOutputRow, TrainError> {
    if model.workers.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut worker_rows = Vec::with_capacity(model.workers.len());
    for worker in &model.workers {
        let cache = mini_transformer_forward_for_attention_and_position(
            worker,
            context_window,
            attention_kind,
            position_policy,
        )?;
        worker_rows.push(cache);
    }

    let logits_q8 = match composition {
        MiniTransformerSwarmComposition::AverageLogits => {
            mini_transformer_average_worker_logits_q8(&worker_rows)
        }
        MiniTransformerSwarmComposition::ConfidenceWeighted => {
            mini_transformer_confidence_weighted_worker_logits_q8(&worker_rows)
        }
        MiniTransformerSwarmComposition::ConfidenceRouter => {
            mini_transformer_confidence_routed_worker_logits_q8(&worker_rows)
        }
    };
    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15).ok_or(TrainError::CoreRejected(
        "mini_transformer_swarm_output_softmax",
    ))?;

    Ok(ByteVocabOutputRow {
        logits_q8,
        probabilities_q15,
    })
}

fn mini_transformer_swarm_prompt_affinity(
    model: &MiniTransformerMlpSwarmModel,
    prompt: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    composition: MiniTransformerSwarmComposition,
    max_windows: usize,
) -> Result<MiniTransformerSwarmPromptAffinityTrace, TrainError> {
    if prompt.len() < 2 || max_windows == 0 {
        return Ok(MiniTransformerSwarmPromptAffinityTrace {
            eval_windows: 0,
            probability_error_q15: 0,
            score: 0,
        });
    }
    if model.context_seq_len == 0
        || model.workers.is_empty()
        || attention_kind.uses_incremental_state()
    {
        return Err(TrainError::InvalidConfig);
    }

    let start = prompt.len().saturating_sub(max_windows.saturating_add(1));
    let mut eval_windows = 0_usize;
    let mut probability_error_q15 = 0_usize;
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for target_index in start.max(1)..prompt.len() {
        let context = &prompt[..target_index];
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let row = mini_transformer_swarm_ensemble_row_for_context(
            model,
            context_window,
            attention_kind,
            position_policy,
            composition,
        )?;
        probability_error_q15 = probability_error_q15.saturating_add(
            byte_sample_probability_error_q15(&row.probabilities_q15, prompt[target_index]),
        );
        eval_windows = eval_windows.saturating_add(1);
    }

    let mean_error = probability_error_q15
        .checked_div(eval_windows)
        .and_then(|value| i64::try_from(value).ok())
        .unwrap_or(i64::from(i16::MAX));
    let score = i64::from(i16::MAX)
        .saturating_sub(mean_error)
        .saturating_mul(10);

    Ok(MiniTransformerSwarmPromptAffinityTrace {
        eval_windows,
        probability_error_q15,
        score,
    })
}

fn mini_transformer_average_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    let mut sums = [0_i64; BYTE_VOCAB];
    for row in rows {
        for (sum, &logit) in sums.iter_mut().zip(row.logits_q8.iter()) {
            *sum = sum.saturating_add(i64::from(logit));
        }
    }
    let divisor = rows.len().max(1) as i64;
    mini_transformer_logit_sums_to_q8(&sums, divisor)
}

fn mini_transformer_confidence_weighted_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    let mut sums = [0_i64; BYTE_VOCAB];
    let mut total_weight = 0_i64;
    for row in rows {
        let weight = i64::from(mini_transformer_logit_margin_q8(&row.logits_q8).max(1));
        total_weight = total_weight.saturating_add(weight);
        for (sum, &logit) in sums.iter_mut().zip(row.logits_q8.iter()) {
            *sum = sum.saturating_add(i64::from(logit).saturating_mul(weight));
        }
    }
    mini_transformer_logit_sums_to_q8(&sums, total_weight.max(1))
}

fn mini_transformer_confidence_routed_worker_logits_q8(
    rows: &[MiniTransformerMlpForwardCache],
) -> [i32; BYTE_VOCAB] {
    rows.iter()
        .enumerate()
        .max_by_key(|&(index, row)| {
            (
                mini_transformer_logit_margin_q8(&row.logits_q8),
                core::cmp::Reverse(index),
            )
        })
        .map(|(_, row)| row.logits_q8)
        .unwrap_or([0_i32; BYTE_VOCAB])
}

fn mini_transformer_logit_sums_to_q8(sums: &[i64; BYTE_VOCAB], divisor: i64) -> [i32; BYTE_VOCAB] {
    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (out, &sum) in logits_q8.iter_mut().zip(sums.iter()) {
        let averaged = sum / divisor.max(1);
        *out = averaged.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    logits_q8
}

fn mini_transformer_logit_margin_q8(logits_q8: &[i32; BYTE_VOCAB]) -> i32 {
    let mut best = i32::MIN;
    let mut second = i32::MIN;
    for &logit in logits_q8 {
        if logit > best {
            second = best;
            best = logit;
        } else if logit > second {
            second = logit;
        }
    }
    best.saturating_sub(second).max(0)
}

pub fn generate_mini_transformer_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_and_priors(
        model,
        prompt,
        config,
        MiniTransformerAttentionKind::Base2Softmax,
        decode_priors,
    )
}

pub fn generate_mini_transformer_with_attention_kind(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        None,
    )
}

pub fn mini_transformer_next_token_row_with_attention_kind_position_policy(
    model: &MiniTransformerMlpModel,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerNextTokenRow, TrainError> {
    let attention_kind = if model.transformer_layers() == 1 {
        attention_kind.preferred_generation_kind(position_policy)
    } else {
        attention_kind
    };
    if attention_kind.uses_incremental_state() {
        return Err(TrainError::InvalidConfig);
    }
    let cache = mini_transformer_forward_for_attention_and_position(
        model,
        context,
        attention_kind,
        position_policy,
    )?;
    Ok(MiniTransformerNextTokenRow {
        logits_q8: cache.logits_q8,
        probabilities_q15: cache.probabilities_q15,
    })
}

pub fn generate_mini_transformer_with_attention_kind_and_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_position_policy_and_priors(
        model,
        prompt,
        config,
        attention_kind,
        MiniTransformerPositionPolicy::LearnedAbsolute,
        decode_priors,
    )
}

pub fn generate_mini_transformer_with_attention_kind_position_policy_and_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
        model,
        prompt,
        config,
        attention_kind,
        position_policy,
        decode_priors,
        DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT,
    )
}

pub fn generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    decode_priors: Option<&ByteDecodePriors>,
    ttt_learning_rate_shift: u8,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    let attention_kind = if model.transformer_layers() == 1 {
        attention_kind.preferred_generation_kind(position_policy)
    } else {
        attention_kind
    };
    if attention_kind == MiniTransformerAttentionKind::LinearStreamingNope {
        return generate_mini_transformer_streaming_linear_nope_with_priors(
            model,
            prompt,
            config,
            decode_priors,
        );
    }
    if attention_kind == MiniTransformerAttentionKind::LinearStreamingTttNope {
        return generate_mini_transformer_streaming_linear_ttt_nope_with_priors(
            model,
            prompt,
            config,
            decode_priors,
            ttt_learning_rate_shift,
        );
    }

    if prompt.is_empty() || model.context_seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut padded_context = Vec::with_capacity(model.context_seq_len);

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let context_len = model.context_seq_len.min(context.len());
        let context_start = context.len() - context_len;
        let context_window = if context_len < model.context_seq_len {
            padded_context.clear();
            padded_context.resize(model.context_seq_len - context_len, b' ');
            padded_context.extend_from_slice(&context[context_start..]);
            padded_context.as_slice()
        } else {
            &context[context_start..]
        };
        let cache = mini_transformer_forward_for_attention_and_position(
            model,
            context_window,
            attention_kind,
            position_policy,
        )?;
        let selection = select_byte_from_row_with_priors(
            &cache.logits_q8,
            &cache.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: cache.logits_q8[predicted_index],
            predicted_probability_q15: cache.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind,
        position_policy,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: None,
        steps,
    })
}

fn generate_mini_transformer_streaming_linear_nope_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    if prompt.is_empty() || model.context_seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut workspace = MiniTransformerStreamingLinearWorkspace::new()?;
    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);

    let mut current_row = None;
    for &token in prompt {
        current_row = Some(mini_transformer_streaming_linear_nope_step(
            model,
            token,
            &mut workspace,
        )?);
    }
    let mut current_row = current_row.ok_or(TrainError::InvalidConfig)?;

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let selection = select_byte_from_row_with_priors(
            &current_row.logits_q8,
            &current_row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: current_row.logits_q8[predicted_index],
            predicted_probability_q15: current_row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });

        if step_index + 1 < config.max_new_tokens {
            current_row = mini_transformer_streaming_linear_nope_step(
                model,
                predicted_token,
                &mut workspace,
            )?;
        }
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind: MiniTransformerAttentionKind::LinearStreamingNope,
        position_policy: MiniTransformerPositionPolicy::Nope,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: None,
        steps,
    })
}

fn generate_mini_transformer_streaming_linear_ttt_nope_with_priors(
    model: &MiniTransformerMlpModel,
    prompt: &[u8],
    config: ByteGenerationConfig,
    decode_priors: Option<&ByteDecodePriors>,
    ttt_learning_rate_shift: u8,
) -> Result<MiniTransformerGenerationTrace, TrainError> {
    if prompt.is_empty() || model.context_seq_len == 0 || ttt_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_decode_priors(config.decode, decode_priors)?;

    let mut workspace = MiniTransformerStreamingLinearWorkspace::new()?;
    let mut context = prompt.to_vec();
    let mut generated_bytes = Vec::with_capacity(config.max_new_tokens);
    let mut steps = Vec::with_capacity(config.max_new_tokens);
    let mut prompt_state_delta_l1 = 0_u64;
    let mut generated_state_delta_l1 = 0_u64;
    let mut zero_delta_count = 0_usize;
    let mut step_count = 0_usize;

    let mut current_row = None;
    for &token in prompt {
        let (row, delta_l1) = mini_transformer_streaming_linear_ttt_nope_step(
            model,
            token,
            &mut workspace,
            ttt_learning_rate_shift,
        )?;
        current_row = Some(row);
        prompt_state_delta_l1 = prompt_state_delta_l1.saturating_add(delta_l1);
        step_count = step_count.saturating_add(1);
        if delta_l1 == 0 {
            zero_delta_count = zero_delta_count.saturating_add(1);
        }
    }
    let mut current_row = current_row.ok_or(TrainError::InvalidConfig)?;

    for step_index in 0..config.max_new_tokens {
        let input_token = *context.last().ok_or(TrainError::InvalidConfig)?;
        let selection = select_byte_from_row_with_priors(
            &current_row.logits_q8,
            &current_row.probabilities_q15,
            config.decode,
            step_index,
            &context,
            decode_priors,
        )?;
        let predicted_token = selection.token;
        let predicted_index = usize::from(predicted_token);
        generated_bytes.push(predicted_token);
        context.push(predicted_token);
        steps.push(ByteGenerationStepTrace {
            step_index,
            input_token,
            predicted_token,
            predicted_logit_q8: current_row.logits_q8[predicted_index],
            predicted_probability_q15: current_row.probabilities_q15[predicted_index],
            candidate_count: selection.candidate_count,
            rejected_candidates: selection.rejected_candidates,
        });

        let (row, delta_l1) = mini_transformer_streaming_linear_ttt_nope_step(
            model,
            predicted_token,
            &mut workspace,
            ttt_learning_rate_shift,
        )?;
        current_row = row;
        generated_state_delta_l1 = generated_state_delta_l1.saturating_add(delta_l1);
        step_count = step_count.saturating_add(1);
        if delta_l1 == 0 {
            zero_delta_count = zero_delta_count.saturating_add(1);
        }
    }

    Ok(MiniTransformerGenerationTrace {
        config,
        attention_kind: MiniTransformerAttentionKind::LinearStreamingTttNope,
        position_policy: MiniTransformerPositionPolicy::Nope,
        prompt_bytes: prompt.to_vec(),
        generated_bytes,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
        context_seq_len: model.context_seq_len,
        decode_priors: decode_priors.map(ByteDecodePriors::trace),
        ttt_stats: Some(MiniTransformerStreamingTttStats {
            learning_rate_shift: ttt_learning_rate_shift,
            step_count,
            zero_delta_count,
            prompt_state_delta_l1,
            generated_state_delta_l1,
            total_state_delta_l1: prompt_state_delta_l1.saturating_add(generated_state_delta_l1),
        }),
        steps,
    })
}

struct MiniTransformerStreamingLinearWorkspace {
    attention_q: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_k: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_v: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_context: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_prediction: [i16; MINI_TRANSFORMER_D_MODEL],
    state_kv: Vec<i64>,
    key_sums: Vec<i64>,
    embedding_output: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_output: [i16; MINI_TRANSFORMER_D_MODEL],
    attention_residual: [i16; MINI_TRANSFORMER_D_MODEL],
    mlp_up: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_gate: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_gated: [i16; MINI_TRANSFORMER_HIDDEN_DIM],
    mlp_output: [i16; MINI_TRANSFORMER_D_MODEL],
    block_output: [i16; MINI_TRANSFORMER_D_MODEL],
}

impl MiniTransformerStreamingLinearWorkspace {
    fn new() -> Result<Self, TrainError> {
        let (state_len, key_sum_len) =
            linear_attention_state_lengths(MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HEADS)
                .ok_or(TrainError::InvalidConfig)?;
        let mut workspace = Self {
            attention_q: [0; MINI_TRANSFORMER_D_MODEL],
            attention_k: [0; MINI_TRANSFORMER_D_MODEL],
            attention_v: [0; MINI_TRANSFORMER_D_MODEL],
            attention_context: [0; MINI_TRANSFORMER_D_MODEL],
            attention_prediction: [0; MINI_TRANSFORMER_D_MODEL],
            state_kv: vec![0; state_len],
            key_sums: vec![0; key_sum_len],
            embedding_output: [0; MINI_TRANSFORMER_D_MODEL],
            attention_output: [0; MINI_TRANSFORMER_D_MODEL],
            attention_residual: [0; MINI_TRANSFORMER_D_MODEL],
            mlp_up: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_gate: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_gated: [0; MINI_TRANSFORMER_HIDDEN_DIM],
            mlp_output: [0; MINI_TRANSFORMER_D_MODEL],
            block_output: [0; MINI_TRANSFORMER_D_MODEL],
        };
        clear_linear_attention_state_checked(
            MINI_TRANSFORMER_D_MODEL,
            MINI_TRANSFORMER_HEADS,
            LinearAttentionState {
                state_kv: &mut workspace.state_kv,
                key_sums: &mut workspace.key_sums,
            },
        )
        .ok_or(TrainError::InvalidConfig)?;
        Ok(workspace)
    }
}

fn mini_transformer_streaming_linear_nope_step(
    model: &MiniTransformerMlpModel,
    token: u8,
    workspace: &mut MiniTransformerStreamingLinearWorkspace,
) -> Result<ByteVocabOutputRow, TrainError> {
    mini_transformer_embedding_token_nope_q15(
        &model.embeddings,
        token,
        &mut workspace.embedding_output,
    )?;

    let attention_params = SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: &model.q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: &model.k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: &model.v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: &model.o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    };
    linear_attention_step_i16_q15_checked(
        &workspace.embedding_output,
        attention_params,
        LinearAttentionStepWorkspace {
            q: &mut workspace.attention_q,
            k: &mut workspace.attention_k,
            v: &mut workspace.attention_v,
            context: &mut workspace.attention_context,
        },
        LinearAttentionState {
            state_kv: &mut workspace.state_kv,
            key_sums: &mut workspace.key_sums,
        },
        &mut workspace.attention_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_attention_step",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.embedding_output,
        &workspace.attention_output,
        &mut workspace.attention_residual,
    )?;

    let mlp_params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: &model.up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: &model.gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: &model.down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    gated_mlp_i16_q15_checked(
        &workspace.attention_residual,
        mlp_params,
        GatedMlpWorkspace {
            up: &mut workspace.mlp_up,
            gate: &mut workspace.mlp_gate,
            gated: &mut workspace.mlp_gated,
        },
        &mut workspace.mlp_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_mlp",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.attention_residual,
        &workspace.mlp_output,
        &mut workspace.block_output,
    )?;
    mini_transformer_output_row_for(&model.output_weights, &workspace.block_output)
}

fn mini_transformer_streaming_linear_ttt_nope_step(
    model: &MiniTransformerMlpModel,
    token: u8,
    workspace: &mut MiniTransformerStreamingLinearWorkspace,
    ttt_learning_rate_shift: u8,
) -> Result<(ByteVocabOutputRow, u64), TrainError> {
    mini_transformer_embedding_token_nope_q15(
        &model.embeddings,
        token,
        &mut workspace.embedding_output,
    )?;

    let attention_params = SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: &model.q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: &model.k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: &model.v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: &model.o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    };
    let delta_l1 = linear_attention_ttt_step_i16_q15_checked(
        &workspace.embedding_output,
        attention_params,
        LinearAttentionTttStepWorkspace {
            q: &mut workspace.attention_q,
            k: &mut workspace.attention_k,
            v: &mut workspace.attention_v,
            context: &mut workspace.attention_context,
            prediction: &mut workspace.attention_prediction,
        },
        LinearAttentionState {
            state_kv: &mut workspace.state_kv,
            key_sums: &mut workspace.key_sums,
        },
        &mut workspace.attention_output,
        ttt_learning_rate_shift,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_ttt_attention_step",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.embedding_output,
        &workspace.attention_output,
        &mut workspace.attention_residual,
    )?;

    let mlp_params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: &model.up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: &model.gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: &model.down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len: 1,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    gated_mlp_i16_q15_checked(
        &workspace.attention_residual,
        mlp_params,
        GatedMlpWorkspace {
            up: &mut workspace.mlp_up,
            gate: &mut workspace.mlp_gate,
            gated: &mut workspace.mlp_gated,
        },
        &mut workspace.mlp_output,
    )
    .ok_or(TrainError::CoreRejected(
        "mini_transformer_streaming_linear_ttt_mlp",
    ))?;

    add_i16_residual_rows_checked(
        &workspace.attention_residual,
        &workspace.mlp_output,
        &mut workspace.block_output,
    )?;
    let row = mini_transformer_output_row_for(&model.output_weights, &workspace.block_output)?;
    Ok((row, delta_l1))
}

fn mini_transformer_embedding_token_nope_q15(
    embeddings: &[i16],
    token: u8,
    output: &mut [i16; MINI_TRANSFORMER_D_MODEL],
) -> Result<(), TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidConfig);
    }
    let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
    let row = embeddings
        .get(row_start..row_start + MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidModel("mini transformer embedding row"))?;
    output.copy_from_slice(row);
    Ok(())
}
