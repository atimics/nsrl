#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerBlockExpertMetrics,
    MiniTransformerBlockExpertObjective, MiniTransformerBlockLowRankExpert,
    MiniTransformerMlpEvalConfig, MiniTransformerMlpModel, MiniTransformerMlpTrainConfig,
    MiniTransformerPositionPolicy, evaluate_mini_transformer_block_expert,
    evaluate_mini_transformer_mlp_windows, mini_transformer_next_token_row_with_block_expert,
    mini_transformer_output_from_hidden_q15, mini_transformer_output_gradient_to_hidden_q15,
    train_mini_transformer_block_expert_with_layer_scope_and_loss_guard,
};

const DEFAULT_PROJECTION_SEED: u64 = 0x424c_4f43_4b45_5850;
const DEFAULT_ROUTER_FEATURE_PROJECTION_SEED: u64 = 0x5254_5250_524f_4a31;
const DEFAULT_ROUTER_FEATURE_PROJECTION_SHIFT: u8 = 4;
const ROUTER_FEATURE_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterFeatureKind {
    Pooled,
    SignedProjection,
}

impl RouterFeatureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pooled => "pooled",
            Self::SignedProjection => "signed_projection",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RouteAggregate {
    windows: usize,
    mistakes: usize,
    probability_error_q15: usize,
    decisions: usize,
    route_switches: usize,
    utilization_tokens: [usize; 3],
}

impl RouteAggregate {
    fn add_group(
        &mut self,
        losses: &[[usize; 3]],
        mistakes: &[[usize; 3]],
        start: usize,
        end: usize,
        expert: usize,
    ) {
        self.decisions = self.decisions.saturating_add(1);
        self.utilization_tokens[expert] =
            self.utilization_tokens[expert].saturating_add(end - start);
        for index in start..end {
            self.windows = self.windows.saturating_add(1);
            self.mistakes = self.mistakes.saturating_add(mistakes[index][expert]);
            self.probability_error_q15 = self
                .probability_error_q15
                .saturating_add(losses[index][expert]);
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-block-expert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("init") => init(args),
        Some("train") => train(args),
        Some("eval") => eval(args),
        Some("signature") => signature(args),
        Some("score") => score(args),
        Some("generate") => generate(args),
        Some("--help" | "-h") => {
            help();
            Ok(())
        }
        _ => Err("expected init, train, eval, signature, score, or generate".into()),
    }
}

fn init(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut expert_out = None;
    let mut trace_out = None;
    let mut rank = 8_usize;
    let mut projection_seed = DEFAULT_PROJECTION_SEED;
    let mut residual_shift = 8_u8;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => expert_out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--rank" => rank = required(&mut args, "--rank")?.parse()?,
            "--projection-seed" => {
                projection_seed = required(&mut args, "--projection-seed")?.parse()?
            }
            "--residual-shift" => {
                residual_shift = required(&mut args, "--residual-shift")?.parse()?
            }
            other => return Err(format!("unknown init argument: {other}").into()),
        }
    }
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let expert = MiniTransformerBlockLowRankExpert::new_for_model_with_residual_shift(
        &model,
        rank,
        projection_seed,
        residual_shift,
    )?;
    let artifact = expert.to_bytes();
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_block_low_rank_expert_init.v1\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"layers\":{},\"rank\":{},\"projection_seed\":{},\"residual_shift\":{},\"parameter_count\":{},\"zero_identity\":true}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            expert.transformer_layers,
            expert.rank,
            expert.projection_seed,
            expert.residual_shift,
            expert.parameter_count(),
        ),
    )?;
    Ok(())
}

fn train(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_out = None;
    let mut trace_out = None;
    let mut resume_path = None;
    let mut epochs = 1_usize;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut batch_windows = 16_usize;
    let mut token_offset = 0_usize;
    let mut learning_rate = 256_i64;
    let mut learning_rate_shift = 0_u8;
    let mut rank = 8_usize;
    let mut projection_seed = DEFAULT_PROJECTION_SEED;
    let mut residual_shift = 8_u8;
    let mut train_layer_spec = String::from("all");
    let mut bidirectional_loss_guard = false;
    let mut objective = MiniTransformerBlockExpertObjective::CrossEntropy;
    let mut attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => expert_out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--resume-expert" => {
                resume_path = Some(PathBuf::from(required(&mut args, "--resume-expert")?))
            }
            "--epochs" => epochs = required(&mut args, "--epochs")?.parse()?,
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--batch-windows" => batch_windows = required(&mut args, "--batch-windows")?.parse()?,
            "--token-offset" => token_offset = required(&mut args, "--token-offset")?.parse()?,
            "--learning-rate" => learning_rate = required(&mut args, "--learning-rate")?.parse()?,
            "--learning-rate-shift" => {
                learning_rate_shift = required(&mut args, "--learning-rate-shift")?.parse()?
            }
            "--rank" => rank = required(&mut args, "--rank")?.parse()?,
            "--projection-seed" => {
                projection_seed = required(&mut args, "--projection-seed")?.parse()?
            }
            "--residual-shift" => {
                residual_shift = required(&mut args, "--residual-shift")?.parse()?
            }
            "--train-layer" => train_layer_spec = required(&mut args, "--train-layer")?,
            "--bidirectional-loss-guard" => bidirectional_loss_guard = true,
            "--objective" => {
                objective = match required(&mut args, "--objective")?.as_str() {
                    "cross-entropy" => MiniTransformerBlockExpertObjective::CrossEntropy,
                    "probability-error" => MiniTransformerBlockExpertObjective::ProbabilityError,
                    _ => {
                        return Err(
                            "--objective requires cross-entropy or probability-error".into()
                        );
                    }
                }
            }
            "--attention" => {
                attention_kind = parse_attention(&required(&mut args, "--attention")?)?
            }
            "--position" => position_policy = parse_position(&required(&mut args, "--position")?)?,
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    let all_tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let tokens = all_tokens
        .get(token_offset..)
        .ok_or("token offset is out of range")?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let mut expert = if let Some(path) = resume_path {
        MiniTransformerBlockLowRankExpert::from_bytes(&fs::read(path)?)?
    } else {
        MiniTransformerBlockLowRankExpert::new_for_model_with_residual_shift(
            &model,
            rank,
            projection_seed,
            residual_shift,
        )?
    };
    expert.validate_for_model(&model)?;
    let train_layer = match train_layer_spec.as_str() {
        "all" => None,
        "final" => Some(expert.transformer_layers - 1),
        value => Some(value.parse::<usize>()?),
    };
    let eval_config = MiniTransformerMlpEvalConfig {
        seq_len: model.context_seq_len,
        stride,
        max_windows,
        attention_kind,
        position_policy,
    };
    let initial = evaluate_mini_transformer_block_expert(tokens, &model, &expert, eval_config)?;
    let train_config = MiniTransformerMlpTrainConfig {
        epochs,
        seq_len: model.context_seq_len,
        stride,
        max_windows,
        batch_windows,
        attention_kind,
        position_policy,
        ..MiniTransformerMlpTrainConfig::default()
    };
    let stats = train_mini_transformer_block_expert_with_layer_scope_and_loss_guard(
        tokens,
        &model,
        &mut expert,
        train_config,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        train_layer,
        bidirectional_loss_guard,
        objective,
    )?;
    let final_metrics =
        evaluate_mini_transformer_block_expert(tokens, &model, &expert, eval_config)?;
    let artifact = expert.to_bytes();
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_block_low_rank_expert_train.v5\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"parameter_count\":{},\"config\":{{\"layers\":{},\"rank\":{},\"projection_seed\":{},\"residual_shift\":{},\"train_layer\":{},\"bidirectional_loss_guard\":{},\"objective\":\"{}\",\"epochs\":{},\"token_offset\":{},\"stride\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate\":{},\"learning_rate_shift\":{},\"attention\":\"{}\",\"position\":\"{}\"}},\"initial\":{},\"final\":{},\"updates\":{{\"optimizer_steps\":{},\"accepted_forward_steps\":{},\"accepted_reverse_steps\":{},\"rejected_steps\":{},\"weight_delta_l1\":{},\"weight_saturation_count\":{},\"hidden_saturation_count\":{}}}}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            expert.parameter_count(),
            expert.transformer_layers,
            expert.rank,
            expert.projection_seed,
            expert.residual_shift,
            train_layer.map_or_else(|| "\"all\"".to_string(), |layer| layer.to_string()),
            bidirectional_loss_guard,
            objective.as_str(),
            epochs,
            token_offset,
            stride,
            optional_usize(max_windows),
            batch_windows,
            learning_rate,
            learning_rate_shift,
            attention_kind.as_str(),
            position_policy.as_str(),
            metrics_json(initial),
            metrics_json(final_metrics),
            stats.optimizer_steps,
            stats.accepted_forward_steps,
            stats.accepted_reverse_steps,
            stats.rejected_steps,
            stats.weight_delta_l1,
            stats.weight_saturation_count,
            stats.hidden_saturation_count,
        ),
    )?;
    Ok(())
}

fn eval(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_path = None;
    let mut trace_out = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_path = Some(PathBuf::from(required(&mut args, "--expert")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--attention" => {
                attention_kind = parse_attention(&required(&mut args, "--attention")?)?
            }
            "--position" => position_policy = parse_position(&required(&mut args, "--position")?)?,
            other => return Err(format!("unknown eval argument: {other}").into()),
        }
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let artifact = fs::read(expert_path.ok_or("--expert is required")?)?;
    let expert = MiniTransformerBlockLowRankExpert::from_bytes(&artifact)?;
    let metrics = evaluate_mini_transformer_block_expert(
        &tokens,
        &model,
        &expert,
        MiniTransformerMlpEvalConfig {
            seq_len: model.context_seq_len,
            stride,
            max_windows,
            attention_kind,
            position_policy,
        },
    )?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_block_low_rank_expert_eval.v2\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"rank\":{},\"residual_shift\":{},\"parameter_count\":{},\"stride\":{},\"max_windows\":{},\"attention\":\"{}\",\"position\":\"{}\",\"metrics\":{}}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            expert.rank,
            expert.residual_shift,
            expert.parameter_count(),
            stride,
            optional_usize(max_windows),
            attention_kind.as_str(),
            position_policy.as_str(),
            metrics_json(metrics),
        ),
    )?;
    Ok(())
}

fn signature(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_path = None;
    let mut model_path = None;
    let mut output_path = None;
    let mut trace_path = None;
    let mut stride = 7_usize;
    let mut max_samples = None;
    let mut attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input_path = Some(PathBuf::from(required(&mut args, "--input")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => output_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_path = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-samples" => max_samples = Some(required(&mut args, "--max-samples")?.parse()?),
            "--attention" => {
                attention_kind = parse_attention(&required(&mut args, "--attention")?)?
            }
            "--position" => position_policy = parse_position(&required(&mut args, "--position")?)?,
            other => return Err(format!("unknown signature argument: {other}").into()),
        }
    }
    if stride == 0 || max_samples == Some(0) {
        return Err("signature bounds must be positive".into());
    }
    let input_path = input_path.ok_or("--input is required")?;
    let input_bytes = fs::read(&input_path)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let output_path = output_path.ok_or("--out is required")?;
    let mut output = BufWriter::new(fs::File::create(&output_path)?);
    writeln!(
        output,
        "sample_id\twindows\tmistakes\tprobability_error_q15\tmean_probability_error_q15\thidden_gradient_signature_q15"
    )?;
    let reader = BufReader::new(input_bytes.as_slice());
    let mut samples = 0_usize;
    let mut windows = 0_usize;
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line_index == 0 && line == "sample_id\tprompt_hex" {
            continue;
        }
        if max_samples.is_some_and(|limit| samples >= limit) {
            break;
        }
        let (sample_id, prompt_hex) = line
            .split_once('\t')
            .ok_or("signature input requires sample_id and prompt_hex")?;
        let tokens = decode_hex(prompt_hex)?;
        let records = evaluate_mini_transformer_mlp_windows(
            &tokens,
            &model,
            MiniTransformerMlpEvalConfig {
                seq_len: model.context_seq_len,
                stride,
                max_windows: None,
                attention_kind,
                position_policy,
            },
        )?;
        if records
            .iter()
            .any(|record| record.invalid_forward_count != 0)
        {
            return Err(format!("invalid frozen-trunk forward for {sample_id}").into());
        }
        let mut signed = [0_i64; 16];
        let mut magnitude = [0_u64; 16];
        let mut sample_mistakes = 0_usize;
        let mut sample_error = 0_usize;
        for record in &records {
            let target = tokens[record.end];
            let row = mini_transformer_output_from_hidden_q15(&model, &record.last_hidden_q15)?;
            sample_mistakes =
                sample_mistakes.saturating_add(usize::from(argmax(&row.logits_q8) != target));
            sample_error = sample_error
                .saturating_add(sample_probability_error(&row.probabilities_q15, target));
            let grad_output = core::array::from_fn(|class| {
                let value = i32::from(row.probabilities_q15[class])
                    - if class == usize::from(target) {
                        i32::from(i16::MAX)
                    } else {
                        0
                    };
                value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
            });
            let grad_hidden = mini_transformer_output_gradient_to_hidden_q15(&model, &grad_output)?;
            for bucket in 0..16 {
                for &gradient in &grad_hidden[bucket * 8..bucket * 8 + 8] {
                    signed[bucket] = signed[bucket].saturating_add(i64::from(gradient));
                    magnitude[bucket] =
                        magnitude[bucket].saturating_add(i64::from(gradient).unsigned_abs());
                }
            }
        }
        let denominator = i64::try_from(records.len().saturating_mul(8))?;
        let mut features = [0_i16; 32];
        for bucket in 0..16 {
            features[bucket] = (signed[bucket] / denominator)
                .clamp(i64::from(i16::MIN), i64::from(i16::MAX))
                as i16;
            features[16 + bucket] = (i64::try_from(magnitude[bucket])? / denominator)
                .clamp(0, i64::from(i16::MAX)) as i16;
        }
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}",
            sample_id,
            records.len(),
            sample_mistakes,
            sample_error,
            sample_error / records.len().max(1),
            features
                .iter()
                .map(i16::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )?;
        samples += 1;
        windows = windows.saturating_add(records.len());
        mistakes = mistakes.saturating_add(sample_mistakes);
        probability_error_q15 = probability_error_q15.saturating_add(sample_error);
    }
    output.flush()?;
    if samples == 0 {
        return Err("signature input contains no samples".into());
    }
    drop(output);
    let output_bytes = fs::read(&output_path)?;
    fs::write(
        trace_path.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_hidden_gradient_signatures.v1\",\"input\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"windows\":{},\"stride\":{}}},\"model\":{{\"hash\":\"0x{:016x}\",\"attention\":\"{}\",\"position\":\"{}\"}},\"signature\":{{\"channels\":32,\"signed_hidden_gradient_channels\":16,\"magnitude_hidden_gradient_channels\":16,\"hash\":\"0x{:016x}\"}},\"metrics\":{{\"mistakes\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{}}}}}\n",
            json_string(&input_path.to_string_lossy()),
            fnv64(&input_bytes),
            samples,
            windows,
            stride,
            model.model_hash(),
            attention_kind.as_str(),
            position_policy.as_str(),
            fnv64(&output_bytes),
            mistakes,
            probability_error_q15,
            probability_error_q15 / windows.max(1),
        ),
    )?;
    Ok(())
}

fn score(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_path = None;
    let mut model_path = None;
    let mut expert_specs = Vec::new();
    let mut output_path = None;
    let mut details_path = None;
    let mut stride = 1_usize;
    let mut span_len = 16_usize;
    let mut max_samples = None;
    let mut router_feature_kind = RouterFeatureKind::Pooled;
    let mut router_feature_projection_seed = DEFAULT_ROUTER_FEATURE_PROJECTION_SEED;
    let mut router_feature_projection_shift = DEFAULT_ROUTER_FEATURE_PROJECTION_SHIFT;
    let mut attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input_path = Some(PathBuf::from(required(&mut args, "--input")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_specs.push(required(&mut args, "--expert")?),
            "--out" => output_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--details-out" => {
                details_path = Some(PathBuf::from(required(&mut args, "--details-out")?))
            }
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--span-len" => span_len = required(&mut args, "--span-len")?.parse()?,
            "--max-samples" => max_samples = Some(required(&mut args, "--max-samples")?.parse()?),
            "--router-features" => {
                router_feature_kind = match required(&mut args, "--router-features")?.as_str() {
                    "pooled" => RouterFeatureKind::Pooled,
                    "signed" | "signed-projection" => RouterFeatureKind::SignedProjection,
                    _ => return Err("--router-features requires pooled or signed".into()),
                }
            }
            "--router-feature-seed" => {
                router_feature_projection_seed =
                    required(&mut args, "--router-feature-seed")?.parse()?
            }
            "--router-feature-shift" => {
                router_feature_projection_shift =
                    required(&mut args, "--router-feature-shift")?.parse()?
            }
            "--attention" => {
                attention_kind = parse_attention(&required(&mut args, "--attention")?)?
            }
            "--position" => position_policy = parse_position(&required(&mut args, "--position")?)?,
            other => return Err(format!("unknown score argument: {other}").into()),
        }
    }
    if expert_specs.len() != 3
        || stride == 0
        || span_len == 0
        || max_samples == Some(0)
        || router_feature_projection_shift > 30
    {
        return Err("score requires three experts and positive bounds".into());
    }
    let input_path = input_path.ok_or("--input is required")?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let mut expert_ids = Vec::new();
    let mut experts = Vec::new();
    let mut expert_hashes = Vec::new();
    for spec in expert_specs {
        let (id, artifact_path) = spec.split_once('=').ok_or("--expert requires ID=PATH")?;
        if id.is_empty() || id.contains(['\t', ',', '=']) {
            return Err("invalid expert ID".into());
        }
        let bytes = fs::read(artifact_path)?;
        let expert = MiniTransformerBlockLowRankExpert::from_bytes(&bytes)?;
        expert.validate_for_model(&model)?;
        if experts
            .first()
            .is_some_and(|first: &MiniTransformerBlockLowRankExpert| {
                first.rank != expert.rank || first.residual_shift != expert.residual_shift
            })
        {
            return Err("scored block experts must share rank and residual shift".into());
        }
        expert_ids.push(id.to_string());
        experts.push(expert);
        expert_hashes.push(fnv64(&bytes));
    }

    let input_bytes = fs::read(&input_path)?;
    let reader = BufReader::new(input_bytes.as_slice());
    let mut details = BufWriter::new(fs::File::create(
        details_path.ok_or("--details-out is required")?,
    )?);
    writeln!(
        details,
        "sample_id\ttarget_offset\ttarget_hex\tchild_probability_error_q15\tchild_mistakes\trouter_hidden_features_q15\tprompt_route\tspan_route\ttoken_route"
    )?;
    let mut fixed = core::array::from_fn::<_, 3, _>(|_| RouteAggregate::default());
    let mut prompt = RouteAggregate::default();
    let mut span = RouteAggregate::default();
    let mut token = RouteAggregate::default();
    let mut sample_count = 0_usize;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line_index == 0 && line == "sample_id\tprompt_hex" {
            continue;
        }
        if max_samples.is_some_and(|limit| sample_count >= limit) {
            break;
        }
        let (sample_id, prompt_hex) = line
            .split_once('\t')
            .ok_or("score input row requires sample_id and prompt_hex")?;
        let tokens = decode_hex(prompt_hex)?;
        let records = evaluate_mini_transformer_mlp_windows(
            &tokens,
            &model,
            MiniTransformerMlpEvalConfig {
                seq_len: model.context_seq_len,
                stride,
                max_windows: None,
                attention_kind,
                position_policy,
            },
        )?;
        let mut losses = vec![[0_usize; 3]; records.len()];
        let mut mistakes = vec![[0_usize; 3]; records.len()];
        for (index, record) in records.iter().enumerate() {
            let target = tokens[record.end];
            let context = &tokens[record.start..record.end];
            for (expert_index, expert) in experts.iter().enumerate().take(3) {
                let row = mini_transformer_next_token_row_with_block_expert(
                    &model,
                    expert,
                    context,
                    attention_kind,
                    position_policy,
                )?;
                losses[index][expert_index] =
                    sample_probability_error(&row.probabilities_q15, target);
                mistakes[index][expert_index] = usize::from(argmax(&row.logits_q8) != target);
            }
        }
        for (expert_index, aggregate) in fixed.iter_mut().enumerate() {
            aggregate.add_group(&losses, &mistakes, 0, records.len(), expert_index);
        }
        let prompt_choice = best_expert(&losses, &mistakes, 0, records.len());
        prompt.add_group(&losses, &mistakes, 0, records.len(), prompt_choice);

        let mut span_routes = vec![0_usize; records.len()];
        let mut previous_span = None;
        for start in (0..records.len()).step_by(span_len) {
            let end = (start + span_len).min(records.len());
            let choice = best_expert(&losses, &mistakes, start, end);
            if previous_span.is_some_and(|previous| previous != choice) {
                span.route_switches = span.route_switches.saturating_add(1);
            }
            previous_span = Some(choice);
            span.add_group(&losses, &mistakes, start, end, choice);
            span_routes[start..end].fill(choice);
        }
        let mut token_routes = vec![0_usize; records.len()];
        let mut previous_token = None;
        for (index, route) in token_routes.iter_mut().enumerate() {
            let choice = best_expert(&losses, &mistakes, index, index + 1);
            if previous_token.is_some_and(|previous| previous != choice) {
                token.route_switches = token.route_switches.saturating_add(1);
            }
            previous_token = Some(choice);
            token.add_group(&losses, &mistakes, index, index + 1, choice);
            *route = choice;
        }
        for (index, record) in records.iter().enumerate() {
            let router_features = match router_feature_kind {
                RouterFeatureKind::Pooled => record.router_hidden_features_q15,
                RouterFeatureKind::SignedProjection => signed_router_features_q15(
                    &record.last_hidden_q15,
                    router_feature_projection_seed,
                    router_feature_projection_shift,
                ),
            };
            writeln!(
                details,
                "{}\t{}\t{:02x}\t{},{},{}\t{},{},{}\t{}\t{}\t{}\t{}",
                sample_id,
                record.end,
                tokens[record.end],
                losses[index][0],
                losses[index][1],
                losses[index][2],
                mistakes[index][0],
                mistakes[index][1],
                mistakes[index][2],
                router_features
                    .iter()
                    .map(i16::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                prompt_choice,
                span_routes[index],
                token_routes[index],
            )?;
        }
        sample_count += 1;
    }
    details.flush()?;
    if sample_count == 0 {
        return Err("score input contains no samples".into());
    }
    let best_fixed = (0..3)
        .min_by_key(|&index| {
            (
                fixed[index].probability_error_q15,
                fixed[index].mistakes,
                index,
            )
        })
        .ok_or("no fixed expert")?;
    let mut output = BufWriter::new(fs::File::create(output_path.ok_or("--out is required")?)?);
    let schema = if router_feature_kind == RouterFeatureKind::Pooled {
        "nsrl.shared_trunk_block_low_rank_routing_ablation.v1"
    } else {
        "nsrl.shared_trunk_block_low_rank_routing_ablation.v2"
    };
    let router_feature_json = if router_feature_kind == RouterFeatureKind::Pooled {
        String::new()
    } else {
        format!(
            ",\"router_features\":{{\"kind\":\"{}\",\"count\":32,\"projection_seed\":{},\"projection_shift\":{}}}",
            router_feature_kind.as_str(),
            router_feature_projection_seed,
            router_feature_projection_shift,
        )
    };
    writeln!(
        output,
        "{{\"schema\":\"{}\",\"dataset\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"windows\":{},\"stride\":{}}},\"trunk\":{{\"model_hash\":\"0x{:016x}\",\"feature_forward_count\":{},\"expert_forward_count\":{}}},\"experts\":{{\"ids\":[{},{},{}],\"artifact_hashes\":[\"0x{:016x}\",\"0x{:016x}\",\"0x{:016x}\"],\"rank\":{},\"residual_shift\":{},\"parameter_count_each\":{}}},\"fixed_experts\":[{},{},{}],\"best_fixed_expert\":{},\"oracle_routes\":{{\"prompt\":{},\"span\":{},\"token\":{}}},\"attention\":\"{}\",\"position\":\"{}\"{},\"known_non_claims\":[\"target_aware_oracle_ceiling\",\"expert_forwards_not_shared_after_first_adapted_block\",\"does_not_claim_language_model_quality\"]}}",
        schema,
        json_string(&input_path.to_string_lossy()),
        fnv64(&input_bytes),
        sample_count,
        prompt.windows,
        stride,
        model.model_hash(),
        prompt.windows,
        prompt.windows.saturating_mul(3),
        json_string(&expert_ids[0]),
        json_string(&expert_ids[1]),
        json_string(&expert_ids[2]),
        expert_hashes[0],
        expert_hashes[1],
        expert_hashes[2],
        experts[0].rank,
        experts[0].residual_shift,
        experts[0].parameter_count(),
        route_json(&fixed[0]),
        route_json(&fixed[1]),
        route_json(&fixed[2]),
        best_fixed,
        route_json(&prompt),
        route_json(&span),
        route_json(&token),
        attention_kind.as_str(),
        position_policy.as_str(),
        router_feature_json,
    )?;
    output.flush()?;
    Ok(())
}

fn generate(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut expert_path = None;
    let mut prompt = Vec::new();
    let mut text_out = None;
    let mut trace_out = None;
    let mut max_new_tokens = 128_usize;
    let mut top_k = 1_usize;
    let mut sample_seed = 1_u64;
    let mut printable_only = false;
    let mut attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_path = Some(PathBuf::from(required(&mut args, "--expert")?)),
            "--prompt" => prompt = required(&mut args, "--prompt")?.into_bytes(),
            "--text-out" => text_out = Some(PathBuf::from(required(&mut args, "--text-out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--max-new-tokens" => {
                max_new_tokens = required(&mut args, "--max-new-tokens")?.parse()?
            }
            "--top-k" => top_k = required(&mut args, "--top-k")?.parse()?,
            "--sample-seed" => sample_seed = required(&mut args, "--sample-seed")?.parse()?,
            "--printable-only" => printable_only = true,
            "--attention" => {
                attention_kind = parse_attention(&required(&mut args, "--attention")?)?
            }
            "--position" => position_policy = parse_position(&required(&mut args, "--position")?)?,
            other => return Err(format!("unknown generate argument: {other}").into()),
        }
    }
    if prompt.is_empty() || top_k == 0 {
        return Err("prompt must be non-empty and top-k positive".into());
    }
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let artifact = fs::read(expert_path.ok_or("--expert is required")?)?;
    let expert = MiniTransformerBlockLowRankExpert::from_bytes(&artifact)?;
    let mut sequence = prompt.clone();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut rng = sample_seed;
    for _ in 0..max_new_tokens {
        let context_len = sequence.len().min(model.context_seq_len);
        let mut context = vec![b' '; model.context_seq_len];
        context[model.context_seq_len - context_len..]
            .copy_from_slice(&sequence[sequence.len() - context_len..]);
        let row = mini_transformer_next_token_row_with_block_expert(
            &model,
            &expert,
            &context,
            attention_kind,
            position_policy,
        )?;
        let token = sample_token(
            &row.logits_q8,
            &row.probabilities_q15,
            printable_only,
            top_k,
            &mut rng,
        );
        sequence.push(token);
        generated.push(token);
    }
    fs::write(text_out.ok_or("--text-out is required")?, &generated)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_block_low_rank_generation.v1\",\"model_hash\":\"0x{:016x}\",\"expert_hash\":\"0x{:016x}\",\"prompt_hex\":\"{}\",\"generated_hex\":\"{}\",\"max_new_tokens\":{},\"top_k\":{},\"sample_seed\":{},\"printable_only\":{},\"attention\":\"{}\",\"position\":\"{}\"}}\n",
            model.model_hash(),
            fnv64(&artifact),
            hex(&prompt),
            hex(&generated),
            max_new_tokens,
            top_k,
            sample_seed,
            printable_only,
            attention_kind.as_str(),
            position_policy.as_str(),
        ),
    )?;
    Ok(())
}

fn sample_token(
    logits: &[i32; 256],
    probabilities: &[i16; 256],
    printable_only: bool,
    top_k: usize,
    rng: &mut u64,
) -> u8 {
    let mut candidates = if printable_only {
        (32_u8..=126).collect::<Vec<_>>()
    } else {
        (u8::MIN..=u8::MAX).collect::<Vec<_>>()
    };
    candidates.sort_by_key(|&token| (core::cmp::Reverse(logits[usize::from(token)]), token));
    candidates.truncate(top_k.min(candidates.len()));
    if candidates.len() == 1 {
        return candidates[0];
    }
    let total = candidates
        .iter()
        .map(|&token| i32::from(probabilities[usize::from(token)]).max(1) as u64)
        .sum::<u64>();
    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut ticket = *rng % total.max(1);
    for token in candidates {
        let weight = i32::from(probabilities[usize::from(token)]).max(1) as u64;
        if ticket < weight {
            return token;
        }
        ticket -= weight;
    }
    b' '
}

fn sample_probability_error(probabilities: &[i16; 256], target: u8) -> usize {
    let target = usize::from(target);
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities[target])).max(0) as usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if index != target {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
}

fn argmax(values: &[i32; 256]) -> u8 {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map_or(0, |(index, _)| index as u8)
}

fn best_expert(losses: &[[usize; 3]], mistakes: &[[usize; 3]], start: usize, end: usize) -> usize {
    (0..3)
        .min_by_key(|&expert| {
            (
                losses[start..end]
                    .iter()
                    .map(|row| row[expert])
                    .sum::<usize>(),
                mistakes[start..end]
                    .iter()
                    .map(|row| row[expert])
                    .sum::<usize>(),
                expert,
            )
        })
        .unwrap_or(0)
}

fn route_json(route: &RouteAggregate) -> String {
    format!(
        "{{\"windows\":{},\"decisions\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{},\"route_switches\":{},\"utilization_tokens\":[{},{},{}]}}",
        route.windows,
        route.decisions,
        route.mistakes,
        route.windows.saturating_sub(route.mistakes) * 1000 / route.windows.max(1),
        route.probability_error_q15,
        route.probability_error_q15 / route.windows.max(1),
        route.route_switches,
        route.utilization_tokens[0],
        route.utilization_tokens[1],
        route.utilization_tokens[2],
    )
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if !value.len().is_multiple_of(2) {
        return Err("hex input length must be even".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| Ok(u8::from_str_radix(&value[index..index + 2], 16)?))
        .collect()
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value.is_control() => output.push_str(&format!("\\u{:04x}", value as u32)),
            value => output.push(value),
        }
    }
    output.push('"');
    output
}

fn metrics_json(metrics: MiniTransformerBlockExpertMetrics) -> String {
    format!(
        "{{\"windows\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{},\"hidden_saturation_count\":{}}}",
        metrics.windows,
        metrics.mistakes,
        metrics.accuracy_per_mille(),
        metrics.probability_error_q15,
        metrics.mean_probability_error_q15(),
        metrics.hidden_saturation_count,
    )
}

fn signed_router_features_q15(
    hidden_q15: &[i16],
    seed: u64,
    projection_shift: u8,
) -> [i16; ROUTER_FEATURE_COUNT] {
    core::array::from_fn(|feature| {
        let sum = hidden_q15
            .iter()
            .enumerate()
            .map(|(channel, &value)| {
                i64::from(value) * router_projection_sign(seed, feature, channel)
            })
            .sum::<i64>();
        let divisor = 1_i64 << projection_shift;
        (sum / divisor).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
    })
}

fn router_projection_sign(seed: u64, feature: usize, channel: usize) -> i64 {
    let mut value = seed
        ^ (feature as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (channel as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    if value & 1 == 0 { -1 } else { 1 }
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 15)]));
    }
    out
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn parse_attention(
    value: &str,
) -> Result<MiniTransformerAttentionKind, Box<dyn std::error::Error>> {
    match value {
        "base2" | "base2-softmax" => Ok(MiniTransformerAttentionKind::Base2Softmax),
        "linear" => Ok(MiniTransformerAttentionKind::Linear),
        _ => Err("--attention requires base2 or linear".into()),
    }
}

fn parse_position(
    value: &str,
) -> Result<MiniTransformerPositionPolicy, Box<dyn std::error::Error>> {
    match value {
        "learned" | "learned-absolute" => Ok(MiniTransformerPositionPolicy::LearnedAbsolute),
        "nope" => Ok(MiniTransformerPositionPolicy::Nope),
        _ => Err("--position requires learned or nope".into()),
    }
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn help() {
    println!(
        "nsrl-mini-transformer-block-expert\n\
         init --model PATH --out PATH --trace PATH [--rank N] [--projection-seed N] [--residual-shift N]\n\
         train --tokens PATH --model PATH --out PATH --trace PATH [--resume-expert PATH] [--rank N] [--residual-shift N] [--train-layer all|final|N] [--bidirectional-loss-guard] [--objective cross-entropy|probability-error] [--epochs N] [--token-offset N] [--stride N] [--max-windows N] [--batch-windows N] [--learning-rate N] [--learning-rate-shift N] [--attention base2|linear] [--position learned|nope]\n\
         eval --tokens PATH --model PATH --expert PATH --trace PATH [--stride N] [--max-windows N] [--attention base2|linear] [--position learned|nope]\n\
         signature --input PATH --model PATH --out PATH --trace PATH [--stride N] [--max-samples N] [--attention base2|linear] [--position learned|nope]\n\
         score --input PATH --model PATH --expert ID=PATH --expert ID=PATH --expert ID=PATH --out PATH --details-out PATH [--stride N] [--span-len N] [--max-samples N] [--router-features pooled|signed] [--router-feature-seed N] [--router-feature-shift N] [--attention base2|linear] [--position learned|nope]\n\
         generate --model PATH --expert PATH --prompt TEXT --text-out PATH --trace PATH [--max-new-tokens N] [--top-k N] [--sample-seed N] [--printable-only] [--attention base2|linear] [--position learned|nope]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_router_projection_is_deterministic_and_retains_within_bucket_signal() {
        let mut hidden = [0_i16; 128];
        hidden[0] = 512;
        hidden[1] = -512;

        let left = signed_router_features_q15(&hidden, 17, 4);
        let right = signed_router_features_q15(&hidden, 17, 4);
        let other_seed = signed_router_features_q15(&hidden, 19, 4);

        assert_eq!(left, right);
        assert!(left.iter().any(|&feature| feature != 0));
        assert_ne!(left, other_seed);
        assert_eq!((i32::from(hidden[0]) + i32::from(hidden[1])) / 2, 0);
    }
}
