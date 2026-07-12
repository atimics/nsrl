#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use nsrl_train::{
    ByteDecodePriors, ByteGenerationConfig, ByteTokenizerId, DecodeStrategy, IntegerAdamConfig,
    MiniTransformerAdamOptimizerState, MiniTransformerAdamTrainScope, MiniTransformerAttentionKind,
    MiniTransformerBatchMode, MiniTransformerBinaryTraceRecord, MiniTransformerBinaryTraceWriter,
    MiniTransformerMlpModel, MiniTransformerMlpSwarmModel, MiniTransformerMlpSwarmTrainConfig,
    MiniTransformerMlpSwarmWorkerArtifact, MiniTransformerMlpTrainConfig,
    MiniTransformerPositionPolicy, MiniTransformerSwarmComposition,
    MiniTransformerSwarmRouteConfig, MiniTransformerSwarmRoutedGenerationExpert,
    MiniTransformerTraceDetail, TrainError, assemble_mini_transformer_mlp_swarm_worker_artifacts,
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors,
    generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift,
    generate_routed_mini_transformer_swarm_experts, route_mini_transformer_swarm_expert_models,
    run_mini_transformer_mlp_integer_adam_training_from_model_with_scope,
    run_mini_transformer_mlp_swarm_scaling_benchmark_from_model,
    run_mini_transformer_mlp_swarm_training_from_model_with_progress,
    run_mini_transformer_mlp_swarm_worker_from_model_with_progress,
    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail,
    run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-train: {error}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceFormat {
    Json,
    Binary,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut mini_transformer_config = MiniTransformerMlpTrainConfig::default();
    let mut integer_adam_config = IntegerAdamConfig::default();
    let mut integer_adam_train_scope = MiniTransformerAdamTrainScope::All;
    let mut enable_rms_norm = false;
    let mut byte_generation_config = ByteGenerationConfig::greedy(32);
    let mut mode = String::from("mini-transformer-mlp");
    let mut trace_format = TraceFormat::Json;
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_paths = Vec::new();
    let mut model_out_path = None;
    let mut optimizer_state_path = None;
    let mut optimizer_state_out_path = None;
    let mut swarm_model_out_path = None;
    let mut manifest_out_path = None;
    let mut prompt = Vec::new();
    let mut trace_path = None;
    let mut progress_path = None;
    let mut progress_interval_batches = 0_usize;
    let mut text_out_path = None;
    let mut generated_only_text = false;
    let mut mini_transformer_attention_kind = MiniTransformerAttentionKind::Linear;
    let mut mini_transformer_position_policy = MiniTransformerPositionPolicy::Nope;
    let mut mini_transformer_ttt_learning_rate_shift =
        nsrl_train::DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT;
    let mut mini_transformer_q_shift_explicit = false;
    let mut mini_transformer_trace_detail = MiniTransformerTraceDetail::Full;
    let mut mini_transformer_trace_detail_explicit = false;
    let mut swarm_workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut swarm_worker_index = None;
    let mut swarm_worker_artifact_out_path = None;
    let mut swarm_worker_artifact_paths = Vec::new();
    let mut swarm_composition = MiniTransformerSwarmComposition::AverageLogits;
    let mut route_config = MiniTransformerSwarmRouteConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                mode = args.next().ok_or(
                    "--mode requires mini-transformer-mlp, mini-transformer-adam, mini-transformer-swarm, mini-transformer-swarm-worker, mini-transformer-swarm-assemble, mini-transformer-swarm-manifest, mini-transformer-swarm-route, mini-transformer-swarm-routed-generate, mini-transformer-swarm-scaling, mini-transformer-swarm-generate, or mini-transformer-generate",
                )?;
            }
            "--epochs" => {
                let epochs = args
                    .next()
                    .ok_or("--epochs requires a following integer")?
                    .parse()?;
                mini_transformer_config.epochs = epochs;
            }
            "--learning-rate" => {
                let value: i32 = args
                    .next()
                    .ok_or("--learning-rate requires a following integer")?
                    .parse()?;
                mini_transformer_config.learning_rate = value;
            }
            "--adam-learning-rate" => {
                integer_adam_config.learning_rate = args
                    .next()
                    .ok_or("--adam-learning-rate requires an integer")?
                    .parse()?;
            }
            "--adam-step-shift" => {
                integer_adam_config.step_shift = args
                    .next()
                    .ok_or("--adam-step-shift requires an integer")?
                    .parse()?;
            }
            "--adam-beta1-shift" => {
                integer_adam_config.beta1_decay_shift = args
                    .next()
                    .ok_or("--adam-beta1-shift requires an integer")?
                    .parse()?;
            }
            "--adam-beta2-shift" => {
                integer_adam_config.beta2_decay_shift = args
                    .next()
                    .ok_or("--adam-beta2-shift requires an integer")?
                    .parse()?;
            }
            "--adam-epsilon" => {
                integer_adam_config.epsilon = args
                    .next()
                    .ok_or("--adam-epsilon requires an integer")?
                    .parse()?;
            }
            "--rms-norm" => {
                enable_rms_norm = true;
            }
            "--adam-train-scope" => {
                integer_adam_train_scope = match args
                    .next()
                    .ok_or(
                        "--adam-train-scope requires all, rms-norm, output, final-mlp, or final-mlp-and-output",
                    )?
                    .as_str()
                {
                    "all" => MiniTransformerAdamTrainScope::All,
                    "rms-norm" => MiniTransformerAdamTrainScope::RmsNorm,
                    "output" => MiniTransformerAdamTrainScope::Output,
                    "final-mlp" => MiniTransformerAdamTrainScope::FinalMlp,
                    "final-mlp-and-output" => MiniTransformerAdamTrainScope::FinalMlpAndOutput,
                    _ => {
                        return Err(
                            "--adam-train-scope requires all, rms-norm, output, final-mlp, or final-mlp-and-output"
                                .into(),
                        );
                    }
                };
            }
            "--lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--lr-shift requires a following integer")?
                    .parse()?;
                mini_transformer_config.output_learning_rate_shift = value;
            }
            "--embed-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--embed-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_config.embedding_learning_rate_shift = value;
            }
            "--mlp-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--mlp-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_config.mlp_learning_rate_shift = value;
            }
            "--attention-lr-shift" => {
                mini_transformer_config.attention_learning_rate_shift = args
                    .next()
                    .ok_or("--attention-lr-shift requires an integer")?
                    .parse()?;
            }
            "--attention-q-lr-shift" => {
                mini_transformer_config.attention_q_learning_rate_shift = args
                    .next()
                    .ok_or("--attention-q-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_q_shift_explicit = true;
            }
            "--attention-qk-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--attention-qk-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_config.attention_qk_learning_rate_shift = value;
                if !mini_transformer_q_shift_explicit {
                    mini_transformer_config.attention_q_learning_rate_shift = value;
                }
            }
            "--adaptive-rule-shifts" => {
                mini_transformer_config.adaptive_rule_shifts = true;
            }
            "--adaptive-rule-interval-batches" => {
                mini_transformer_config.adaptive_rule_interval_batches = args
                    .next()
                    .ok_or("--adaptive-rule-interval-batches requires an integer")?
                    .parse()?;
            }
            "--adaptive-attention-shifts" => {
                mini_transformer_config.adaptive_rule_shifts = true;
                mini_transformer_config.adaptive_attention_shifts = true;
                mini_transformer_config.adaptive_holographic_shifts = true;
            }
            "--adaptive-holographic-shifts" => {
                mini_transformer_config.adaptive_holographic_shifts = true;
            }
            "--swarm-workers" => {
                swarm_workers = args
                    .next()
                    .ok_or("--swarm-workers requires an integer")?
                    .parse()?;
            }
            "--swarm-worker-count" => {
                swarm_workers = args
                    .next()
                    .ok_or("--swarm-worker-count requires an integer")?
                    .parse()?;
            }
            "--swarm-worker-index" => {
                swarm_worker_index = Some(
                    args.next()
                        .ok_or("--swarm-worker-index requires an integer")?
                        .parse()?,
                );
            }
            "--swarm-composition" => {
                swarm_composition = parse_swarm_composition(
                    &args
                        .next()
                        .ok_or("--swarm-composition requires average, confidence-weighted, or confidence-router")?,
                )?;
            }
            "--route-capability" => {
                route_config.required_capabilities.push(
                    args.next()
                        .ok_or("--route-capability requires a capability tag")?,
                );
            }
            "--route-max-artifact-bytes" => {
                route_config.max_artifact_bytes = Some(
                    args.next()
                        .ok_or("--route-max-artifact-bytes requires an integer")?
                        .parse()?,
                );
            }
            "--route-max-parameter-bytes" => {
                route_config.max_parameter_bytes = Some(
                    args.next()
                        .ok_or("--route-max-parameter-bytes requires an integer")?
                        .parse()?,
                );
            }
            "--route-active-experts" => {
                route_config.active_expert_limit = args
                    .next()
                    .ok_or("--route-active-experts requires an integer")?
                    .parse()?;
            }
            "--route-prompt-affinity" => {
                route_config.prompt_affinity = true;
            }
            "--route-prompt-affinity-windows" => {
                route_config.prompt_affinity_max_windows = args
                    .next()
                    .ok_or("--route-prompt-affinity-windows requires an integer")?
                    .parse()?;
            }
            "--attention-vo-error-feedback" => {
                mini_transformer_config.attention_vo_error_feedback = true;
            }
            "--attention-vo-oracle" => {
                mini_transformer_config.attention_vo_oracle = true;
            }
            "--reject-loss-regression" => {
                mini_transformer_config.reject_loss_regression = true;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(
                    args.next().ok_or("--tokens requires a following path")?,
                ));
            }
            "--model" | "--resume-from" => {
                model_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--model/--resume-from requires a following path")?,
                ));
            }
            "--expert" => {
                expert_paths.push(PathBuf::from(
                    args.next().ok_or("--expert requires a following path")?,
                ));
            }
            "--model-out" => {
                model_out_path = Some(PathBuf::from(
                    args.next().ok_or("--model-out requires a following path")?,
                ));
            }
            "--optimizer-state" => {
                optimizer_state_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--optimizer-state requires a following path")?,
                ));
            }
            "--optimizer-state-out" => {
                optimizer_state_out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--optimizer-state-out requires a following path")?,
                ));
            }
            "--swarm-model-out" => {
                swarm_model_out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--swarm-model-out requires a following path")?,
                ));
            }
            "--swarm-worker-out" => {
                swarm_worker_artifact_out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--swarm-worker-out requires a following path")?,
                ));
            }
            "--swarm-worker-artifact" => {
                swarm_worker_artifact_paths.push(PathBuf::from(
                    args.next()
                        .ok_or("--swarm-worker-artifact requires a following path")?,
                ));
            }
            "--manifest-out" => {
                manifest_out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--manifest-out requires a following path")?,
                ));
            }
            "--prompt" => {
                prompt = args
                    .next()
                    .ok_or("--prompt requires a following string")?
                    .into_bytes();
            }
            "--max-new-tokens" => {
                let value = args
                    .next()
                    .ok_or("--max-new-tokens requires an integer")?
                    .parse()?;
                byte_generation_config.max_new_tokens = value;
            }
            "--decode" => {
                let value = args.next().ok_or("--decode requires greedy or sample")?;
                let strategy = match value.as_str() {
                    "greedy" => DecodeStrategy::Greedy,
                    "sample" => DecodeStrategy::Sample,
                    _ => return Err("--decode requires greedy or sample".into()),
                };
                byte_generation_config.decode.strategy = strategy;
            }
            "--sample-seed" => {
                let value = args
                    .next()
                    .ok_or("--sample-seed requires an integer")?
                    .parse()?;
                byte_generation_config.decode.sample_seed = value;
            }
            "--top-k" => {
                let value = args.next().ok_or("--top-k requires an integer")?.parse()?;
                byte_generation_config.decode.top_k = value;
            }
            "--tokenizer" => {
                let value = args
                    .next()
                    .ok_or("--tokenizer requires identity or ascii-lower")?;
                let tokenizer_id = match value.as_str() {
                    "identity" | "byte_identity_u8_v1" => ByteTokenizerId::Identity,
                    "ascii-lower" | "byte_ascii_lower_text_u8_v1" => {
                        ByteTokenizerId::AsciiLowerText
                    }
                    _ => return Err("--tokenizer requires identity or ascii-lower".into()),
                };
                mini_transformer_config.tokenizer_id = tokenizer_id;
                byte_generation_config.tokenizer_id = tokenizer_id;
            }
            "--mini-transformer-attention" => {
                let value = args
                    .next()
                    .ok_or(
                        "--mini-transformer-attention requires base2-softmax, linear, linear-streaming, or linear-streaming-ttt",
                    )?;
                mini_transformer_attention_kind = parse_mini_transformer_attention_kind(&value)?;
                mini_transformer_config.attention_kind = mini_transformer_attention_kind;
            }
            "--mini-transformer-position" => {
                let value = args
                    .next()
                    .ok_or("--mini-transformer-position requires learned-absolute or nope")?;
                mini_transformer_position_policy = parse_mini_transformer_position_policy(&value)?;
                mini_transformer_config.position_policy = mini_transformer_position_policy;
            }
            "--mini-transformer-ttt-lr-shift" => {
                mini_transformer_ttt_learning_rate_shift = args
                    .next()
                    .ok_or("--mini-transformer-ttt-lr-shift requires an integer")?
                    .parse()?;
            }
            "--printable-only" => {
                byte_generation_config.decode.printable_only = true;
            }
            "--ascii-lower-only" => {
                byte_generation_config.decode.ascii_lower_only = true;
            }
            "--repeat-window" => {
                let value = args
                    .next()
                    .ok_or("--repeat-window requires an integer")?
                    .parse()?;
                byte_generation_config.decode.repeat_window = value;
            }
            "--repeat-penalty-shift" => {
                let value = args
                    .next()
                    .ok_or("--repeat-penalty-shift requires an integer")?
                    .parse()?;
                byte_generation_config.decode.repeat_penalty_shift = value;
            }
            "--max-repeat-run" => {
                let value = args
                    .next()
                    .ok_or("--max-repeat-run requires an integer")?
                    .parse()?;
                byte_generation_config.decode.max_repeat_run = value;
            }
            "--no-repeat-ngram" => {
                let value = args
                    .next()
                    .ok_or("--no-repeat-ngram requires an integer")?
                    .parse()?;
                byte_generation_config.decode.no_repeat_ngram_order = value;
            }
            "--corpus-prior" => {
                byte_generation_config.decode.corpus_prior = true;
            }
            "--corpus-prior-logit-shift" => {
                let value = args
                    .next()
                    .ok_or("--corpus-prior-logit-shift requires an integer")?
                    .parse()?;
                byte_generation_config.decode.corpus_prior_logit_shift = value;
            }
            "--strict-adjacency" => {
                byte_generation_config.decode.strict_adjacency = true;
            }
            "--seq-len" => {
                let seq_len = args
                    .next()
                    .ok_or("--seq-len requires an integer")?
                    .parse()?;
                mini_transformer_config.seq_len = seq_len;
            }
            "--stride" => {
                let stride = args.next().ok_or("--stride requires an integer")?.parse()?;
                mini_transformer_config.stride = stride;
            }
            "--window-offset" => {
                let window_offset = args
                    .next()
                    .ok_or("--window-offset requires an integer")?
                    .parse()?;
                mini_transformer_config.window_offset = window_offset;
            }
            "--batch-windows" => {
                let value = args
                    .next()
                    .ok_or("--batch-windows requires an integer")?
                    .parse()?;
                mini_transformer_config.batch_windows = value;
            }
            "--mini-transformer-batch-mode" => {
                mini_transformer_config.batch_mode = parse_mini_transformer_batch_mode(
                    &args
                        .next()
                        .ok_or("--mini-transformer-batch-mode requires serial or map-reduce")?,
                )?;
            }
            "--mini-transformer-map-reduce-workers" => {
                mini_transformer_config.map_reduce_workers = args
                    .next()
                    .ok_or("--mini-transformer-map-reduce-workers requires an integer")?
                    .parse()?;
            }
            "--max-windows" => {
                let max_windows = Some(
                    args.next()
                        .ok_or("--max-windows requires an integer")?
                        .parse()?,
                );
                mini_transformer_config.max_windows = max_windows;
            }
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next().ok_or("--trace requires a following path")?,
                ));
            }
            "--trace-format" => {
                trace_format = parse_trace_format(
                    &args
                        .next()
                        .ok_or("--trace-format requires json or binary")?,
                )?;
            }
            "--mini-transformer-trace-detail" => {
                mini_transformer_trace_detail =
                    parse_mini_transformer_trace_detail(&args.next().ok_or(
                        "--mini-transformer-trace-detail requires full, summary, or none",
                    )?)?;
                mini_transformer_trace_detail_explicit = true;
            }
            "--progress-out" => {
                progress_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--progress-out requires a following path")?,
                ));
            }
            "--progress-interval-batches" => {
                progress_interval_batches = args
                    .next()
                    .ok_or("--progress-interval-batches requires an integer")?
                    .parse()?;
            }
            "--text-out" => {
                text_out_path = Some(PathBuf::from(
                    args.next().ok_or("--text-out requires a following path")?,
                ));
            }
            "--generated-only" => {
                generated_only_text = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    if trace_format == TraceFormat::Binary
        && !matches!(
            mode.as_str(),
            "mini-transformer-mlp" | "mini_transformer_mlp"
        )
    {
        return Err("--trace-format binary currently supports mini-transformer-mlp mode".into());
    }

    let mut binary_output = None;
    let mut trace_output_written = false;
    let line = match mode.as_str() {
        "mini-transformer-adam" | "mini_transformer_adam" => {
            if trace_format == TraceFormat::Binary {
                return Err("integer Adam mode currently emits its versioned JSON trace".into());
            }
            if progress_path.is_some() {
                return Err(
                    "integer Adam mode does not yet emit incremental progress files".into(),
                );
            }
            let path = tokens_path.ok_or("--tokens is required for mini-transformer-adam mode")?;
            let tokens = fs::read(path)?;
            let mut model = if let Some(path) = model_path {
                MiniTransformerMlpModel::from_bytes(&fs::read(path)?)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            if enable_rms_norm {
                model.enable_rms_norm()?;
            }
            let optimizer_state = if let Some(path) = optimizer_state_path {
                Some(MiniTransformerAdamOptimizerState::from_bytes(&fs::read(
                    path,
                )?)?)
            } else {
                None
            };
            let run = run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
                &tokens,
                mini_transformer_config,
                integer_adam_config,
                model,
                optimizer_state,
                integer_adam_train_scope,
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            if let Some(path) = optimizer_state_out_path {
                fs::write(path, run.optimizer_state.try_to_bytes()?)?;
            }
            run.trace.to_json_line()
        }
        "mini-transformer-mlp" | "mini_transformer_mlp" => {
            if mini_transformer_attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
                || mini_transformer_attention_kind
                    == MiniTransformerAttentionKind::LinearStreamingTttNope
            {
                return Err(
                    "--mini-transformer-attention linear-streaming modes are generation-only; train with linear for the rescanned linear-attention backward"
                        .into(),
                );
            }
            let path = tokens_path.ok_or("--tokens is required for mini-transformer-mlp mode")?;
            let tokens = fs::read(path)?;
            let effective_trace_detail =
                if trace_format == TraceFormat::Binary && !mini_transformer_trace_detail_explicit {
                    MiniTransformerTraceDetail::Summary
                } else {
                    mini_transformer_trace_detail
                };
            let model = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                MiniTransformerMlpModel::from_bytes(&model_bytes)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            let progress_interval = if progress_path.is_some() {
                progress_interval_batches.max(1)
            } else {
                0
            };
            let run = if let (TraceFormat::Binary, Some(trace_path)) = (trace_format, &trace_path) {
                let trace_file = fs::File::create(trace_path)?;
                let mut binary_writer =
                    MiniTransformerBinaryTraceWriter::new(BufWriter::new(trace_file));
                let run = {
                    let mut write_progress =
                        |progress: &nsrl_train::MiniTransformerMlpTrainingProgressTrace| {
                            if let Some(path) = progress_path.as_ref() {
                                write_progress_trace(path, &progress.to_json_line())
                            } else {
                                Ok(())
                            }
                        };
                    let mut write_binary_trace = |record: MiniTransformerBinaryTraceRecord<'_>| {
                        binary_writer
                            .write_record(record)
                            .map_err(|_| TrainError::TraceWrite)
                    };
                    run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace(
                    &tokens,
                    mini_transformer_config,
                    model,
                        progress_interval,
                    effective_trace_detail,
                        &mut write_progress,
                        &mut write_binary_trace,
                    )?
                };
                binary_writer.into_inner().flush()?;
                trace_output_written = true;
                run
            } else {
                let mut write_progress =
                    |progress: &nsrl_train::MiniTransformerMlpTrainingProgressTrace| {
                        if let Some(path) = progress_path.as_ref() {
                            write_progress_trace(path, &progress.to_json_line())
                        } else {
                            Ok(())
                        }
                    };
                run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
                    &tokens,
                    mini_transformer_config,
                    model,
                    progress_interval,
                    effective_trace_detail,
                    &mut write_progress,
                )?
            };
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            if trace_format == TraceFormat::Binary && !trace_output_written {
                binary_output = Some(run.trace.to_binary_trace_v1());
                String::new()
            } else if trace_output_written {
                String::new()
            } else {
                run.trace.to_json_line()
            }
        }
        "mini-transformer-swarm" | "mini_transformer_swarm" => {
            if mini_transformer_attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
                || mini_transformer_attention_kind
                    == MiniTransformerAttentionKind::LinearStreamingTttNope
            {
                return Err(
                    "--mini-transformer-attention linear-streaming modes are generation-only; swarm training uses trainable attention modes"
                        .into(),
                );
            }
            let path = tokens_path.ok_or("--tokens is required for mini-transformer-swarm mode")?;
            let tokens = fs::read(path)?;
            let model = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                MiniTransformerMlpModel::from_bytes(&model_bytes)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            let effective_trace_detail = if mini_transformer_trace_detail_explicit {
                mini_transformer_trace_detail
            } else {
                MiniTransformerTraceDetail::None
            };
            let progress_interval = if progress_path.is_some() {
                progress_interval_batches.max(1)
            } else {
                0
            };
            let mut write_progress =
                |progress: &nsrl_train::MiniTransformerMlpSwarmTrainingProgressTrace| {
                    if let Some(path) = progress_path.as_ref() {
                        write_progress_trace(path, &progress.to_json_line())
                    } else {
                        Ok(())
                    }
                };
            let run = run_mini_transformer_mlp_swarm_training_from_model_with_progress(
                &tokens,
                mini_transformer_config,
                MiniTransformerMlpSwarmTrainConfig {
                    workers: swarm_workers,
                    trace_detail: effective_trace_detail,
                },
                model,
                progress_interval,
                &mut write_progress,
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            if let Some(path) = swarm_model_out_path {
                fs::write(path, run.swarm_model.try_to_bytes()?)?;
            }
            if let Some(path) = manifest_out_path {
                fs::write(path, run.swarm_model.to_expert_manifest()?.to_json_line())?;
            }
            run.trace.to_json_line()
        }
        "mini-transformer-swarm-worker" | "mini_transformer_swarm_worker" => {
            if trace_format == TraceFormat::Binary {
                return Err(
                    "--trace-format binary is not supported for mini-transformer-swarm-worker"
                        .into(),
                );
            }
            if mini_transformer_attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
                || mini_transformer_attention_kind
                    == MiniTransformerAttentionKind::LinearStreamingTttNope
            {
                return Err(
                    "--mini-transformer-attention linear-streaming modes are generation-only; swarm worker training uses trainable attention modes"
                        .into(),
                );
            }
            let worker_index =
                swarm_worker_index.ok_or("--swarm-worker-index is required for worker mode")?;
            let path =
                tokens_path.ok_or("--tokens is required for mini-transformer-swarm-worker mode")?;
            let tokens = fs::read(path)?;
            let model = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                MiniTransformerMlpModel::from_bytes(&model_bytes)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            let effective_trace_detail = if mini_transformer_trace_detail_explicit {
                mini_transformer_trace_detail
            } else {
                MiniTransformerTraceDetail::None
            };
            let progress_interval = if progress_path.is_some() {
                progress_interval_batches.max(1)
            } else {
                0
            };
            let mut write_progress =
                |progress: &nsrl_train::MiniTransformerMlpSwarmTrainingProgressTrace| {
                    if let Some(path) = progress_path.as_ref() {
                        write_progress_trace(path, &progress.to_json_line())
                    } else {
                        Ok(())
                    }
                };
            let run = run_mini_transformer_mlp_swarm_worker_from_model_with_progress(
                &tokens,
                mini_transformer_config,
                worker_index,
                swarm_workers,
                model,
                progress_interval,
                effective_trace_detail,
                &mut write_progress,
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.artifact.model.try_to_bytes()?)?;
            }
            if let Some(path) = swarm_worker_artifact_out_path {
                fs::write(path, run.artifact.try_to_bytes()?)?;
            }
            run.artifact.to_json_line()
        }
        "mini-transformer-swarm-assemble" | "mini_transformer_swarm_assemble" => {
            if trace_format == TraceFormat::Binary {
                return Err(
                    "--trace-format binary is not supported for mini-transformer-swarm-assemble"
                        .into(),
                );
            }
            if swarm_worker_artifact_paths.is_empty() {
                return Err(
                    "--swarm-worker-artifact is required for mini-transformer-swarm-assemble"
                        .into(),
                );
            }
            let path = tokens_path
                .ok_or("--tokens is required for mini-transformer-swarm-assemble mode")?;
            let tokens = fs::read(path)?;
            let model = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                MiniTransformerMlpModel::from_bytes(&model_bytes)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            let artifacts = swarm_worker_artifact_paths
                .iter()
                .map(|path| {
                    let bytes = fs::read(path)?;
                    MiniTransformerMlpSwarmWorkerArtifact::from_bytes(&bytes)
                        .map_err(Box::<dyn std::error::Error>::from)
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let run = assemble_mini_transformer_mlp_swarm_worker_artifacts(
                &tokens,
                mini_transformer_config,
                &model,
                artifacts,
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            if let Some(path) = swarm_model_out_path {
                fs::write(path, run.swarm_model.try_to_bytes()?)?;
            }
            if let Some(path) = manifest_out_path {
                fs::write(path, run.swarm_model.to_expert_manifest()?.to_json_line())?;
            }
            run.trace.to_json_line()
        }
        "mini-transformer-swarm-manifest" | "mini_transformer_swarm_manifest" => {
            let path =
                model_path.ok_or("--model is required for mini-transformer-swarm-manifest mode")?;
            let model_bytes = fs::read(path)?;
            let model = MiniTransformerMlpSwarmModel::from_bytes(&model_bytes)?;
            let manifest = model.to_expert_manifest()?.to_json_line();
            if let Some(path) = manifest_out_path {
                fs::write(path, &manifest)?;
            }
            manifest
        }
        "mini-transformer-swarm-route" | "mini_transformer_swarm_route" => {
            if let Some(path) = model_path {
                expert_paths.push(path);
            }
            if expert_paths.is_empty() {
                return Err(
                    "--expert or --model is required for mini-transformer-swarm-route mode".into(),
                );
            }
            if route_config.prompt_affinity
                && mini_transformer_attention_kind.uses_incremental_state()
            {
                return Err(
                    "--mini-transformer-attention streaming modes are not supported for prompt-affinity swarm routing"
                        .into(),
                );
            }
            let mut experts = Vec::with_capacity(expert_paths.len());
            for path in &expert_paths {
                let model_bytes = fs::read(path)?;
                let model = MiniTransformerMlpSwarmModel::from_bytes(&model_bytes)?;
                experts.push(MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: path.to_string_lossy().into_owned(),
                    model,
                });
            }
            let route = route_mini_transformer_swarm_expert_models(
                &experts,
                route_config,
                &prompt,
                mini_transformer_attention_kind,
                mini_transformer_position_policy,
                swarm_composition,
            )?;
            route.to_json_line()
        }
        "mini-transformer-swarm-routed-generate"
        | "mini_transformer_swarm_routed_generate"
        | "mini-transformer-swarm-route-generate"
        | "mini_transformer_swarm_route_generate" => {
            if let Some(path) = model_path {
                expert_paths.push(path);
            }
            if expert_paths.is_empty() {
                return Err(
                    "--expert or --model is required for mini-transformer-swarm-routed-generate mode"
                        .into(),
                );
            }
            if prompt.is_empty() {
                return Err(
                    "--prompt is required for mini-transformer-swarm-routed-generate mode".into(),
                );
            }
            if mini_transformer_attention_kind.uses_incremental_state() {
                return Err(
                    "--mini-transformer-attention streaming modes are not supported for routed swarm generation"
                        .into(),
                );
            }
            let mut experts = Vec::with_capacity(expert_paths.len());
            for path in &expert_paths {
                let model_bytes = fs::read(path)?;
                let model = MiniTransformerMlpSwarmModel::from_bytes(&model_bytes)?;
                experts.push(MiniTransformerSwarmRoutedGenerationExpert {
                    expert_id: path.to_string_lossy().into_owned(),
                    model,
                });
            }
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let routed_generation = generate_routed_mini_transformer_swarm_experts(
                &experts,
                route_config,
                &prompt,
                byte_generation_config,
                mini_transformer_attention_kind,
                mini_transformer_position_policy,
                swarm_composition,
                decode_priors.as_ref(),
            )?;
            write_text_generation(
                &text_out_path,
                &routed_generation.generation.prompt_bytes,
                &routed_generation.generation.generated_bytes,
                generated_only_text,
            )?;
            routed_generation.to_json_line()
        }
        "mini-transformer-swarm-scaling"
        | "mini_transformer_swarm_scaling"
        | "mini-transformer-swarm-bench"
        | "mini_transformer_swarm_bench" => {
            if mini_transformer_attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
                || mini_transformer_attention_kind
                    == MiniTransformerAttentionKind::LinearStreamingTttNope
            {
                return Err(
                    "--mini-transformer-attention linear-streaming modes are generation-only; swarm scaling uses trainable attention modes"
                        .into(),
                );
            }
            if progress_path.is_some() {
                return Err(
                    "--progress-out is not supported for mini-transformer-swarm-scaling".into(),
                );
            }
            if model_out_path.is_some() || swarm_model_out_path.is_some() {
                return Err("--model-out and --swarm-model-out are not supported for mini-transformer-swarm-scaling".into());
            }
            if manifest_out_path.is_some() {
                return Err(
                    "--manifest-out is not supported for mini-transformer-swarm-scaling".into(),
                );
            }
            let path = tokens_path
                .ok_or("--tokens is required for mini-transformer-swarm-scaling mode")?;
            let tokens = fs::read(path)?;
            let model = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                MiniTransformerMlpModel::from_bytes(&model_bytes)?
            } else {
                MiniTransformerMlpModel::new_initial_with_seq_len(mini_transformer_config.seq_len)
            };
            let effective_trace_detail = if mini_transformer_trace_detail_explicit {
                mini_transformer_trace_detail
            } else {
                MiniTransformerTraceDetail::None
            };
            let trace = run_mini_transformer_mlp_swarm_scaling_benchmark_from_model(
                &tokens,
                mini_transformer_config,
                swarm_workers,
                effective_trace_detail,
                model,
            )?;
            trace.to_json_line()
        }
        "mini-transformer-swarm-generate" | "mini_transformer_swarm_generate" => {
            let path =
                model_path.ok_or("--model is required for mini-transformer-swarm-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for mini-transformer-swarm-generate mode".into());
            }
            if mini_transformer_attention_kind.uses_incremental_state() {
                return Err(
                    "--mini-transformer-attention streaming modes are not supported for swarm generation"
                        .into(),
                );
            }
            let model_bytes = fs::read(path)?;
            let model = MiniTransformerMlpSwarmModel::from_bytes(&model_bytes)?;
            if let Some(path) = manifest_out_path {
                fs::write(path, model.to_expert_manifest()?.to_json_line())?;
            }
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let generation =
                generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors(
                    &model,
                    &prompt,
                    byte_generation_config,
                    mini_transformer_attention_kind,
                    mini_transformer_position_policy,
                    swarm_composition,
                    decode_priors.as_ref(),
                )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
                generated_only_text,
            )?;
            generation.to_json_line()
        }
        "mini-transformer-generate" | "mini_transformer_generate" => {
            let path =
                model_path.ok_or("--model is required for mini-transformer-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for mini-transformer-generate mode".into());
            }
            let model_bytes = fs::read(path)?;
            let model = MiniTransformerMlpModel::from_bytes(&model_bytes)?;
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let generation =
                generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
                    &model,
                    &prompt,
                    byte_generation_config,
                    mini_transformer_attention_kind,
                    mini_transformer_position_policy,
                    decode_priors.as_ref(),
                    mini_transformer_ttt_learning_rate_shift,
                )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
                generated_only_text,
            )?;
            generation.to_json_line()
        }
        other => return Err(format!("unknown mode: {other}").into()),
    };
    if let Some(bytes) = binary_output {
        if let Some(path) = trace_path {
            fs::write(path, bytes)?;
        } else {
            io::stdout().write_all(&bytes)?;
        }
    } else if trace_output_written {
        // The mini-transformer binary trace path streams directly during training.
    } else if let Some(path) = trace_path {
        fs::write(path, line)?;
    } else {
        io::stdout().write_all(line.as_bytes())?;
    }

    Ok(())
}

fn print_help() {
    println!(
        "Usage: nsrl-train [--mode mini-transformer-mlp|mini-transformer-adam|mini-transformer-swarm|mini-transformer-swarm-worker|mini-transformer-swarm-assemble|mini-transformer-swarm-manifest|mini-transformer-swarm-route|mini-transformer-swarm-routed-generate|mini-transformer-swarm-scaling|mini-transformer-swarm-generate|mini-transformer-generate] [--tokens PATH] [--model PATH|--resume-from PATH] [--model-out PATH] [--optimizer-state PATH] [--optimizer-state-out PATH] [--adam-learning-rate N] [--adam-step-shift N] [--adam-beta1-shift N] [--adam-beta2-shift N] [--adam-epsilon N] [--adam-train-scope all|output|final-mlp|final-mlp-and-output] [--rms-norm] [--expert PATH] [--swarm-model-out PATH] [--swarm-worker-out PATH] [--swarm-worker-artifact PATH] [--manifest-out PATH] [--prompt TEXT] [--max-new-tokens N] [--decode greedy|sample] [--sample-seed N] [--top-k N] [--tokenizer identity|ascii-lower] [--mini-transformer-attention base2-softmax|linear|linear-streaming|linear-streaming-ttt] [--mini-transformer-position learned-absolute|nope] [--mini-transformer-ttt-lr-shift N] [--printable-only] [--ascii-lower-only] [--repeat-window N] [--repeat-penalty-shift N] [--max-repeat-run N] [--no-repeat-ngram N] [--corpus-prior] [--corpus-prior-logit-shift N] [--strict-adjacency] [--epochs N] [--learning-rate N] [--lr-shift N] [--mlp-lr-shift N] [--embed-lr-shift N] [--attention-lr-shift N] [--attention-q-lr-shift N] [--attention-qk-lr-shift N] [--adaptive-rule-shifts] [--adaptive-rule-interval-batches N] [--adaptive-attention-shifts] [--adaptive-holographic-shifts] [--swarm-workers N|--swarm-worker-count N] [--swarm-worker-index N] [--swarm-composition average|confidence-weighted|confidence-router] [--route-capability TAG] [--route-max-artifact-bytes N] [--route-max-parameter-bytes N] [--route-active-experts N] [--route-prompt-affinity] [--route-prompt-affinity-windows N] [--attention-vo-error-feedback] [--attention-vo-oracle] [--reject-loss-regression] [--seq-len N] [--stride N] [--window-offset N] [--batch-windows N] [--mini-transformer-batch-mode serial|map-reduce] [--mini-transformer-map-reduce-workers N] [--max-windows N] [--trace PATH] [--trace-format json|binary] [--mini-transformer-trace-detail full|summary|none] [--progress-out PATH] [--progress-interval-batches N] [--text-out PATH] [--generated-only]"
    );
    println!("Adam scopes also include rms-norm for internal i16 gamma-only training.");
    println!();
    println!("Runs deterministic mini-transformer training or generation traces.");
}

fn write_progress_trace(path: &PathBuf, line: &str) -> Result<(), TrainError> {
    let mut tmp = path.clone();
    let extension = match path.extension().and_then(|value| value.to_str()) {
        Some(extension) => format!("{extension}.tmp"),
        None => String::from("tmp"),
    };
    tmp.set_extension(extension);
    fs::write(&tmp, line).map_err(|_| TrainError::CoreRejected("progress_write"))?;
    fs::rename(&tmp, path).map_err(|_| TrainError::CoreRejected("progress_rename"))?;
    Ok(())
}

fn parse_mini_transformer_attention_kind(
    value: &str,
) -> Result<MiniTransformerAttentionKind, Box<dyn std::error::Error>> {
    match value {
        "base2-softmax" | "base2_softmax" | "softmax" => {
            Ok(MiniTransformerAttentionKind::Base2Softmax)
        }
        "linear" | "linear-attention" | "linear_attention" => {
            Ok(MiniTransformerAttentionKind::Linear)
        }
        "linear-streaming"
        | "linear_streaming"
        | "linear-incremental"
        | "linear_incremental"
        | "linear-streaming-nope"
        | "linear_streaming_nope" => Ok(MiniTransformerAttentionKind::LinearStreamingNope),
        "linear-streaming-ttt"
        | "linear_streaming_ttt"
        | "linear-streaming-ttt-nope"
        | "linear_streaming_ttt_nope" => Ok(MiniTransformerAttentionKind::LinearStreamingTttNope),
        _ => Err(
            "--mini-transformer-attention requires base2-softmax, linear, linear-streaming, or linear-streaming-ttt"
                .into(),
        ),
    }
}

fn parse_mini_transformer_position_policy(
    value: &str,
) -> Result<MiniTransformerPositionPolicy, Box<dyn std::error::Error>> {
    match value {
        "learned-absolute" | "learned_absolute" | "absolute" | "position-embedding"
        | "position_embedding" => Ok(MiniTransformerPositionPolicy::LearnedAbsolute),
        "nope" | "no-position" | "no_position" | "none" => Ok(MiniTransformerPositionPolicy::Nope),
        _ => Err("--mini-transformer-position requires learned-absolute or nope".into()),
    }
}

fn parse_mini_transformer_batch_mode(
    value: &str,
) -> Result<MiniTransformerBatchMode, Box<dyn std::error::Error>> {
    match value {
        "serial" => Ok(MiniTransformerBatchMode::Serial),
        "map-reduce" | "map_reduce" | "mapreduce" => Ok(MiniTransformerBatchMode::MapReduce),
        _ => Err("--mini-transformer-batch-mode requires serial or map-reduce".into()),
    }
}

fn parse_trace_format(value: &str) -> Result<TraceFormat, Box<dyn std::error::Error>> {
    match value {
        "json" | "jsonl" => Ok(TraceFormat::Json),
        "binary" | "bin" | "nsrlt" | "nsrltrace" => Ok(TraceFormat::Binary),
        _ => Err("--trace-format requires json or binary".into()),
    }
}

fn parse_mini_transformer_trace_detail(
    value: &str,
) -> Result<MiniTransformerTraceDetail, Box<dyn std::error::Error>> {
    match value {
        "full" => Ok(MiniTransformerTraceDetail::Full),
        "summary" | "sampled" | "sample" => Ok(MiniTransformerTraceDetail::Summary),
        "none" | "off" => Ok(MiniTransformerTraceDetail::None),
        _ => Err("--mini-transformer-trace-detail requires full, summary, or none".into()),
    }
}

fn parse_swarm_composition(
    value: &str,
) -> Result<MiniTransformerSwarmComposition, Box<dyn std::error::Error>> {
    match value {
        "average" | "average-logits" | "average_logits" => {
            Ok(MiniTransformerSwarmComposition::AverageLogits)
        }
        "confidence-weighted" | "confidence_weighted" | "weighted" => {
            Ok(MiniTransformerSwarmComposition::ConfidenceWeighted)
        }
        "confidence-router" | "confidence_router" | "router" => {
            Ok(MiniTransformerSwarmComposition::ConfidenceRouter)
        }
        _ => Err(
            "--swarm-composition requires average, confidence-weighted, or confidence-router"
                .into(),
        ),
    }
}

fn load_decode_priors(
    tokens_path: &Option<PathBuf>,
    config: ByteGenerationConfig,
) -> Result<Option<ByteDecodePriors>, Box<dyn std::error::Error>> {
    if !config.decode.corpus_prior && !config.decode.strict_adjacency {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with --corpus-prior or --strict-adjacency")?;
    let tokens = fs::read(path)?;
    Ok(Some(ByteDecodePriors::from_tokens(&tokens)?))
}

fn write_text_generation(
    path: &Option<PathBuf>,
    prompt: &[u8],
    generated: &[u8],
    generated_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let capacity = if generated_only {
            generated.len()
        } else {
            prompt.len() + generated.len()
        };
        let mut text = Vec::with_capacity(capacity);
        if !generated_only {
            text.extend_from_slice(prompt);
        }
        text.extend_from_slice(generated);
        fs::write(path, text)?;
    }
    Ok(())
}
