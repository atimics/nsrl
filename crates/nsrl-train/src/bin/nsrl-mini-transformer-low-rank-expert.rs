#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_core::{round_shift_rhu_i64, saturate_i16};
use nsrl_train::{
    MINI_TRANSFORMER_D_MODEL, MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig,
    MiniTransformerMlpModel, MiniTransformerMlpWindowEvalRecord, MiniTransformerPositionPolicy,
    evaluate_mini_transformer_mlp_windows, mini_transformer_output_from_hidden_q15,
    mini_transformer_output_gradient_to_hidden_q15,
};

const MAGIC_V1: &[u8; 8] = b"NSRLLE1\n";
const MAGIC_V2: &[u8; 8] = b"NSRLLE2\n";
const HIDDEN_EXPERT_MAGIC: &[u8; 8] = b"NSRLHE1\n";
const VOCAB: usize = 256;
const MAX_EXPERT_RANK: usize = MINI_TRANSFORMER_D_MODEL;
const DEFAULT_PROJECTION_SEED: u64 = 0x4c4f_5752_414e_4b31;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LowRankExpert {
    trunk_model_hash: u64,
    rank: usize,
    projection_seed: u64,
    base_gains_q15: [i16; MINI_TRANSFORMER_D_MODEL],
    expansion_weights_q15: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Metrics {
    windows: usize,
    mistakes: usize,
    probability_error_q15: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrainStats {
    optimizer_steps: usize,
    weight_delta_l1: u64,
    weight_saturation_count: usize,
    hidden_saturation_count: usize,
    residual_carry_event_count: usize,
    final_residual_carry_nonzero_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UpdateFingerprint {
    element_count: usize,
    nonzero_count: usize,
    delta_l1: u64,
    delta_linf: u64,
    hash: u64,
}

impl UpdateFingerprint {
    fn to_json(self) -> String {
        format!(
            "{{\"element_count\":{},\"nonzero_count\":{},\"zero_count\":{},\"delta_l1\":{},\"delta_linf\":{},\"hash\":\"0x{:016x}\"}}",
            self.element_count,
            self.nonzero_count,
            self.element_count.saturating_sub(self.nonzero_count),
            self.delta_l1,
            self.delta_linf,
            self.hash,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
        eprintln!("nsrl-mini-transformer-low-rank-expert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => run_train(args),
        Some("eval") => run_eval(args),
        Some("score") => run_score(args),
        Some("generate") => run_generate(args),
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        _ => Err("expected train, eval, score, or generate".into()),
    }
}

fn run_train(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_out = None;
    let mut trace_out = None;
    let mut epochs = 1_usize;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut batch_windows = usize::MAX;
    let mut learning_rate = 1024_i64;
    let mut learning_rate_shift = 0_u8;
    let mut rank = 4_usize;
    let mut projection_seed = DEFAULT_PROJECTION_SEED;
    let mut base_hidden_expert_path = None;
    let mut teacher_model_path = None;
    let mut token_offset = 0_usize;
    let mut resume_expert_path = None;
    let mut error_feedback = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => expert_out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--epochs" => epochs = required(&mut args, "--epochs")?.parse()?,
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--token-offset" => token_offset = required(&mut args, "--token-offset")?.parse()?,
            "--batch-windows" => batch_windows = required(&mut args, "--batch-windows")?.parse()?,
            "--learning-rate" => learning_rate = required(&mut args, "--learning-rate")?.parse()?,
            "--learning-rate-shift" => {
                learning_rate_shift = required(&mut args, "--learning-rate-shift")?.parse()?
            }
            "--rank" => rank = required(&mut args, "--rank")?.parse()?,
            "--projection-seed" => {
                projection_seed = required(&mut args, "--projection-seed")?.parse()?
            }
            "--base-hidden-expert" => {
                base_hidden_expert_path =
                    Some(PathBuf::from(required(&mut args, "--base-hidden-expert")?))
            }
            "--teacher-model" => {
                teacher_model_path = Some(PathBuf::from(required(&mut args, "--teacher-model")?))
            }
            "--resume-expert" => {
                resume_expert_path = Some(PathBuf::from(required(&mut args, "--resume-expert")?))
            }
            "--no-error-feedback" => error_feedback = false,
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    if epochs == 0
        || stride == 0
        || batch_windows == 0
        || learning_rate_shift > 30
        || !(1..=16_384).contains(&learning_rate)
        || rank == 0
        || rank > MAX_EXPERT_RANK
    {
        return Err("invalid training configuration".into());
    }
    let token_bytes = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let tokens = token_bytes
        .get(token_offset..)
        .ok_or("token offset is out of range")?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let records = evaluate_records(tokens, &model, stride, max_windows)?;
    let teacher = if let Some(path) = teacher_model_path {
        let teacher = MiniTransformerMlpModel::from_bytes(&fs::read(path)?)?;
        if teacher.context_seq_len != model.context_seq_len {
            return Err("teacher and trunk context lengths differ".into());
        }
        Some(teacher)
    } else {
        None
    };
    let teacher_probabilities = teacher
        .as_ref()
        .map(|teacher| teacher_probabilities(tokens, teacher, stride, max_windows, &records))
        .transpose()?;
    let base_gains_q15 = if let Some(path) = base_hidden_expert_path {
        read_hidden_expert_gains(&fs::read(path)?, model.model_hash())?
    } else {
        [0; MINI_TRANSFORMER_D_MODEL]
    };
    let (mut expert, resumed_from_hash) = if let Some(path) = resume_expert_path {
        let bytes = fs::read(path)?;
        let expert = LowRankExpert::from_bytes(&bytes)?;
        if expert.trunk_model_hash != model.model_hash() {
            return Err("resumed expert is bound to a different trunk".into());
        }
        (expert, Some(fnv64(&bytes)))
    } else {
        (
            LowRankExpert::new_with_base(
                model.model_hash(),
                rank,
                projection_seed,
                base_gains_q15,
            )?,
            None,
        )
    };
    rank = expert.rank;
    projection_seed = expert.projection_seed;
    let initial_expert = expert.clone();
    let initial = evaluate_expert(&records, tokens, &model, &expert)?;
    let initial_teacher_probability_l1 = teacher_probabilities
        .as_deref()
        .map(|probabilities| {
            evaluate_teacher_probability_l1(&records, &model, &expert, probabilities)
        })
        .transpose()?;
    let stats = train_expert(
        &records,
        tokens,
        &model,
        &mut expert,
        epochs,
        batch_windows,
        learning_rate,
        learning_rate_shift,
        error_feedback,
        teacher_probabilities.as_deref(),
    )?;
    let parameter_update = update_fingerprint(
        &initial_expert.expansion_weights_q15,
        &expert.expansion_weights_q15,
    )?;
    let functional_update = functional_update_fingerprint(&records, &initial_expert, &expert)?;
    let active_rank = active_update_rank(&initial_expert, &expert)?;
    let final_metrics = evaluate_expert(&records, tokens, &model, &expert)?;
    let final_teacher_probability_l1 = teacher_probabilities
        .as_deref()
        .map(|probabilities| {
            evaluate_teacher_probability_l1(&records, &model, &expert, probabilities)
        })
        .transpose()?;
    let artifact = expert.to_bytes()?;
    let objective = if teacher.is_some() {
        "teacher_cross_entropy"
    } else {
        "probability_error"
    };
    let teacher_json = teacher.as_ref().map_or_else(
        || "null".to_string(),
        |teacher| {
            let initial_l1 = initial_teacher_probability_l1.unwrap_or(0);
            let final_l1 = final_teacher_probability_l1.unwrap_or(0);
            format!(
                "{{\"model_hash\":\"0x{:016x}\",\"initial_probability_l1\":{},\"initial_mean_probability_l1\":{},\"final_probability_l1\":{},\"final_mean_probability_l1\":{}}}",
                teacher.model_hash(),
                initial_l1,
                initial_l1 / records.len().max(1),
                final_l1,
                final_l1 / records.len().max(1),
            )
        },
    );
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_low_rank_expert_train.v4\",\"expert_type\":{},\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"resumed_from_hash\":{},\"parameter_count\":{},\"trainable_parameter_count\":{},\"config\":{{\"rank\":{},\"projection_seed\":{},\"epochs\":{},\"token_offset\":{},\"stride\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate\":{},\"learning_rate_shift\":{},\"error_feedback\":{},\"objective\":{}}},\"teacher\":{},\"initial\":{},\"final\":{},\"updates\":{{\"optimizer_steps\":{},\"weight_delta_l1\":{},\"weight_saturation_count\":{},\"hidden_saturation_count\":{},\"residual_carry_event_count\":{},\"final_residual_carry_nonzero_count\":{},\"active_rank\":{},\"parameter_update\":{},\"functional_update\":{}}}}}\n",
            json_string(if expert.has_base_gains() {
                "diagonal_plus_fixed_projection_low_rank_hidden_residual_q15"
            } else {
                "fixed_projection_low_rank_hidden_residual_q15"
            }),
            expert.trunk_model_hash,
            fnv64(&artifact),
            resumed_from_hash
                .map_or_else(|| "null".to_string(), |hash| format!("\"0x{hash:016x}\"")),
            expert.parameter_count(),
            expert.expansion_weights_q15.len(),
            rank,
            projection_seed,
            epochs,
            token_offset,
            stride,
            optional_usize_json(max_windows),
            batch_windows.min(records.len()),
            learning_rate,
            learning_rate_shift,
            error_feedback,
            json_string(objective),
            teacher_json,
            metrics_json(initial),
            metrics_json(final_metrics),
            stats.optimizer_steps,
            stats.weight_delta_l1,
            stats.weight_saturation_count,
            stats.hidden_saturation_count,
            stats.residual_carry_event_count,
            stats.final_residual_carry_nonzero_count,
            active_rank,
            parameter_update.to_json(),
            functional_update.to_json(),
        ),
    )?;
    Ok(())
}

fn run_eval(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_path = None;
    let mut trace_out = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_path = Some(PathBuf::from(required(&mut args, "--expert")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            other => return Err(format!("unknown eval argument: {other}").into()),
        }
    }
    if stride == 0 {
        return Err("stride must be positive".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let expert_bytes = fs::read(expert_path.ok_or("--expert is required")?)?;
    let expert = LowRankExpert::from_bytes(&expert_bytes)?;
    if expert.trunk_model_hash != model.model_hash() {
        return Err("low-rank expert is bound to a different trunk".into());
    }
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let metrics = evaluate_expert(&records, &tokens, &model, &expert)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_low_rank_expert_eval.v2\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"rank\":{},\"has_base_gains\":{},\"parameter_count\":{},\"stride\":{},\"max_windows\":{},\"metrics\":{}}}\n",
            model.model_hash(),
            fnv64(&expert_bytes),
            expert.rank,
            expert.has_base_gains(),
            expert.parameter_count(),
            stride,
            optional_usize_json(max_windows),
            metrics_json(metrics),
        ),
    )?;
    Ok(())
}

fn run_generate(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut expert_path = None;
    let mut prompt = None;
    let mut max_new_tokens = 128_usize;
    let mut text_out = None;
    let mut trace_out = None;
    let mut printable_only = false;
    let mut generated_only = false;
    let mut top_k = 1_usize;
    let mut sample_seed = 7_u64;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_path = Some(PathBuf::from(required(&mut args, "--expert")?)),
            "--prompt" => prompt = Some(required(&mut args, "--prompt")?),
            "--max-new-tokens" => {
                max_new_tokens = required(&mut args, "--max-new-tokens")?.parse()?
            }
            "--top-k" => top_k = required(&mut args, "--top-k")?.parse()?,
            "--sample-seed" => sample_seed = required(&mut args, "--sample-seed")?.parse()?,
            "--text-out" => text_out = Some(PathBuf::from(required(&mut args, "--text-out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--printable-only" => printable_only = true,
            "--generated-only" => generated_only = true,
            other => return Err(format!("unknown generate argument: {other}").into()),
        }
    }
    if max_new_tokens == 0 || top_k == 0 || top_k > VOCAB {
        return Err("max new tokens and top-k must be positive; top-k must be <= 256".into());
    }
    let model_bytes = fs::read(model_path.ok_or("--model is required")?)?;
    let model = MiniTransformerMlpModel::from_bytes(&model_bytes)?;
    let expert_bytes = fs::read(expert_path.ok_or("--expert is required")?)?;
    let expert = LowRankExpert::from_bytes(&expert_bytes)?;
    if expert.trunk_model_hash != model.model_hash() {
        return Err("low-rank expert is bound to a different trunk".into());
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let prompt_bytes = prompt.to_ascii_lowercase().into_bytes();
    let mut sequence = prompt_bytes.clone();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut rng_state = sample_seed;
    for _ in 0..max_new_tokens {
        let next = generate_next_token(
            &sequence,
            &model,
            &expert,
            printable_only,
            top_k,
            &mut rng_state,
        )?;
        sequence.push(next);
        generated.push(next);
    }
    let output = if generated_only {
        &generated
    } else {
        &sequence
    };
    fs::write(text_out.ok_or("--text-out is required")?, output)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_low_rank_generation.v1\",\"model_hash\":\"0x{:016x}\",\"expert_hash\":\"0x{:016x}\",\"prompt_hex\":\"{}\",\"generated_hex\":\"{}\",\"max_new_tokens\":{},\"decode\":{},\"top_k\":{},\"sample_seed\":{},\"printable_only\":{},\"generated_only\":{}}}\n",
            model.model_hash(),
            fnv64(&expert_bytes),
            encode_hex(&prompt_bytes),
            encode_hex(&generated),
            max_new_tokens,
            json_string(if top_k == 1 { "greedy" } else { "top_k_sample" }),
            top_k,
            sample_seed,
            printable_only,
            generated_only,
        ),
    )?;
    Ok(())
}

fn run_score(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_path = None;
    let mut model_path = None;
    let mut expert_specs = Vec::new();
    let mut output_path = None;
    let mut details_path = None;
    let mut stride = 1_usize;
    let mut span_len = 16_usize;
    let mut max_samples = None;
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
            other => return Err(format!("unknown score argument: {other}").into()),
        }
    }
    if expert_specs.len() != 3 || stride == 0 || span_len == 0 || max_samples == Some(0) {
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
        let expert = LowRankExpert::from_bytes(&bytes)?;
        if expert.trunk_model_hash != model.model_hash() {
            return Err("low-rank expert is bound to a different trunk".into());
        }
        if experts.first().is_some_and(|first: &LowRankExpert| {
            first.rank != expert.rank || first.projection_seed != expert.projection_seed
        }) {
            return Err("scored low-rank experts must share rank and projection".into());
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
    let mut fixed = [
        RouteAggregate::default(),
        RouteAggregate::default(),
        RouteAggregate::default(),
    ];
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
        let records = evaluate_records(&tokens, &model, stride, None)?;
        let mut losses = vec![[0_usize; 3]; records.len()];
        let mut mistakes = vec![[0_usize; 3]; records.len()];
        for (index, record) in records.iter().enumerate() {
            let target = tokens[record.end];
            for (expert_index, expert) in experts.iter().enumerate().take(3) {
                let (hidden, _, _) = adapted_hidden(&record.last_hidden_q15, expert);
                let row = mini_transformer_output_from_hidden_q15(&model, &hidden)?;
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
                record
                    .router_hidden_features_q15
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
    writeln!(
        output,
        "{{\"schema\":\"nsrl.shared_trunk_low_rank_routing_ablation.v1\",\"dataset\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"windows\":{},\"stride\":{}}},\"trunk\":{{\"model_hash\":\"0x{:016x}\",\"forward_count\":{},\"naive_three_model_forward_count\":{}}},\"experts\":{{\"ids\":[{},{},{}],\"artifact_hashes\":[\"0x{:016x}\",\"0x{:016x}\",\"0x{:016x}\"],\"rank\":{},\"parameter_count_each\":{}}},\"fixed_experts\":[{},{},{}],\"best_fixed_expert\":{},\"oracle_routes\":{{\"prompt\":{},\"span\":{},\"token\":{}}},\"known_non_claims\":[\"target_aware_oracle_ceiling\",\"fixed_projection_low_rank_residual_not_full_ffn_expert\",\"does_not_claim_language_model_quality\"]}}",
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
        experts[0].parameter_count(),
        route_json(&fixed[0]),
        route_json(&fixed[1]),
        route_json(&fixed[2]),
        best_fixed,
        route_json(&prompt),
        route_json(&span),
        route_json(&token),
    )?;
    output.flush()?;
    Ok(())
}

fn print_help() {
    println!("Usage:");
    println!(
        "  nsrl-mini-transformer-low-rank-expert train --tokens PATH --model PATH --out PATH --trace PATH [--teacher-model PATH] [--resume-expert PATH] [--base-hidden-expert PATH] [--rank N] [--projection-seed N] [--epochs N] [--token-offset N] [--stride N] [--max-windows N] [--batch-windows N] [--learning-rate N] [--learning-rate-shift N] [--no-error-feedback]"
    );
    println!(
        "  nsrl-mini-transformer-low-rank-expert eval --tokens PATH --model PATH --expert PATH --trace PATH [--stride N] [--max-windows N]"
    );
    println!(
        "  nsrl-mini-transformer-low-rank-expert score --input PATH --model PATH --expert ID=PATH --expert ID=PATH --expert ID=PATH --out PATH --details-out PATH [--stride N] [--span-len N] [--max-samples N]"
    );
    println!(
        "  nsrl-mini-transformer-low-rank-expert generate --model PATH --expert PATH --prompt TEXT --text-out PATH --trace PATH [--max-new-tokens N] [--top-k N] [--sample-seed N] [--printable-only] [--generated-only]"
    );
}

fn evaluate_records(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    stride: usize,
    max_windows: Option<usize>,
) -> Result<Vec<MiniTransformerMlpWindowEvalRecord>, Box<dyn std::error::Error>> {
    Ok(evaluate_mini_transformer_mlp_windows(
        tokens,
        model,
        MiniTransformerMlpEvalConfig {
            seq_len: model.context_seq_len,
            stride,
            max_windows,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
        },
    )?)
}

impl LowRankExpert {
    fn new_with_base(
        trunk_model_hash: u64,
        rank: usize,
        projection_seed: u64,
        base_gains_q15: [i16; MINI_TRANSFORMER_D_MODEL],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let count = MINI_TRANSFORMER_D_MODEL
            .checked_mul(rank)
            .ok_or("low-rank parameter overflow")?;
        Ok(Self {
            trunk_model_hash,
            rank,
            projection_seed,
            base_gains_q15,
            expansion_weights_q15: vec![0; count],
        })
    }

    fn has_base_gains(&self) -> bool {
        self.base_gains_q15.iter().any(|&gain| gain != 0)
    }

    fn parameter_count(&self) -> usize {
        self.expansion_weights_q15.len()
            + if self.has_base_gains() {
                MINI_TRANSFORMER_D_MODEL
            } else {
                0
            }
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.rank == 0
            || self.rank > MAX_EXPERT_RANK
            || self.expansion_weights_q15.len() != MINI_TRANSFORMER_D_MODEL * self.rank
        {
            return Err("invalid low-rank expert shape".into());
        }
        Ok(())
    }

    fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC_V2);
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        out.extend_from_slice(&(self.rank as u32).to_le_bytes());
        out.extend_from_slice(&self.projection_seed.to_le_bytes());
        for &value in &self.base_gains_q15 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.expansion_weights_q15 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = fnv64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < MAGIC_V1.len() + 8 + 4 + 8 + 8 {
            return Err("invalid low-rank artifact".into());
        }
        let version = if &bytes[..MAGIC_V1.len()] == MAGIC_V1 {
            1
        } else if &bytes[..MAGIC_V2.len()] == MAGIC_V2 {
            2
        } else {
            return Err("invalid low-rank artifact".into());
        };
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(bytes[checksum_offset..].try_into()?);
        if fnv64(&bytes[..checksum_offset]) != expected_checksum {
            return Err("low-rank expert checksum mismatch".into());
        }
        let mut offset = MAGIC_V1.len();
        let trunk_model_hash = read_u64(bytes, &mut offset)?;
        let rank = read_u32(bytes, &mut offset)? as usize;
        let projection_seed = read_u64(bytes, &mut offset)?;
        let mut base_gains_q15 = [0_i16; MINI_TRANSFORMER_D_MODEL];
        if version == 2 {
            for value in &mut base_gains_q15 {
                *value = read_i16(bytes, &mut offset)?;
            }
        }
        let count = MINI_TRANSFORMER_D_MODEL
            .checked_mul(rank)
            .ok_or("low-rank parameter overflow")?;
        if offset + count * 2 != checksum_offset {
            return Err("low-rank artifact length mismatch".into());
        }
        let mut expansion_weights_q15 = Vec::with_capacity(count);
        for _ in 0..count {
            expansion_weights_q15.push(read_i16(bytes, &mut offset)?);
        }
        let expert = Self {
            trunk_model_hash,
            rank,
            projection_seed,
            base_gains_q15,
            expansion_weights_q15,
        };
        expert.validate()?;
        Ok(expert)
    }
}

fn projection_sign(seed: u64, rank_index: usize, dim: usize) -> i64 {
    let mut value = seed
        ^ (rank_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (dim as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    if value & 1 == 0 { 1 } else { -1 }
}

fn latent_q15(hidden: &[i16; MINI_TRANSFORMER_D_MODEL], rank: usize, seed: u64) -> Vec<i16> {
    (0..rank)
        .map(|rank_index| {
            let sum = hidden
                .iter()
                .enumerate()
                .map(|(dim, &value)| i64::from(value) * projection_sign(seed, rank_index, dim))
                .sum::<i64>();
            saturate_i16(round_shift_rhu_i64(sum, 7))
        })
        .collect()
}

fn adapted_hidden(
    base: &[i16; MINI_TRANSFORMER_D_MODEL],
    expert: &LowRankExpert,
) -> ([i16; MINI_TRANSFORMER_D_MODEL], Vec<i16>, usize) {
    let mut diagonal = [0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut saturation_count = 0_usize;
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        let diagonal_residual = round_shift_rhu_i64(
            i64::from(base[dim]).saturating_mul(i64::from(expert.base_gains_q15[dim])),
            15,
        );
        let raw = i64::from(base[dim]).saturating_add(diagonal_residual);
        diagonal[dim] = saturate_i16(raw);
        saturation_count += usize::from(i64::from(diagonal[dim]) != raw);
    }
    let latent = latent_q15(&diagonal, expert.rank, expert.projection_seed);
    let mut output = [0_i16; MINI_TRANSFORMER_D_MODEL];
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        let residual_acc = (0..expert.rank)
            .map(|rank_index| {
                i64::from(latent[rank_index])
                    * i64::from(expert.expansion_weights_q15[dim * expert.rank + rank_index])
            })
            .sum::<i64>();
        let residual = round_shift_rhu_i64(residual_acc, 15);
        let raw = i64::from(diagonal[dim]).saturating_add(residual);
        output[dim] = saturate_i16(raw);
        saturation_count += usize::from(i64::from(output[dim]) != raw);
    }
    (output, latent, saturation_count)
}

fn update_fingerprint(
    before: &[i16],
    after: &[i16],
) -> Result<UpdateFingerprint, Box<dyn std::error::Error>> {
    if before.len() != after.len() {
        return Err("update fingerprint shape mismatch".into());
    }
    let mut hash = 0xcbf29ce484222325_u64;
    let mut nonzero_count = 0_usize;
    let mut delta_l1 = 0_u64;
    let mut delta_linf = 0_u64;
    for (&before, &after) in before.iter().zip(after) {
        let delta = i64::from(after) - i64::from(before);
        let magnitude = delta.unsigned_abs();
        nonzero_count = nonzero_count.saturating_add(usize::from(delta != 0));
        delta_l1 = delta_l1.saturating_add(magnitude);
        delta_linf = delta_linf.max(magnitude);
        for byte in delta.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(UpdateFingerprint {
        element_count: before.len(),
        nonzero_count,
        delta_l1,
        delta_linf,
        hash,
    })
}

fn functional_update_fingerprint(
    records: &[MiniTransformerMlpWindowEvalRecord],
    before: &LowRankExpert,
    after: &LowRankExpert,
) -> Result<UpdateFingerprint, Box<dyn std::error::Error>> {
    let mut before_hidden = Vec::with_capacity(records.len() * MINI_TRANSFORMER_D_MODEL);
    let mut after_hidden = Vec::with_capacity(records.len() * MINI_TRANSFORMER_D_MODEL);
    for record in records {
        before_hidden.extend_from_slice(&adapted_hidden(&record.last_hidden_q15, before).0);
        after_hidden.extend_from_slice(&adapted_hidden(&record.last_hidden_q15, after).0);
    }
    update_fingerprint(&before_hidden, &after_hidden)
}

fn active_update_rank(
    before: &LowRankExpert,
    after: &LowRankExpert,
) -> Result<usize, Box<dyn std::error::Error>> {
    if before.rank != after.rank
        || before.expansion_weights_q15.len() != after.expansion_weights_q15.len()
    {
        return Err("active rank shape mismatch".into());
    }
    Ok((0..after.rank)
        .filter(|&rank_index| {
            (0..MINI_TRANSFORMER_D_MODEL).any(|dim| {
                let index = dim * after.rank + rank_index;
                before.expansion_weights_q15[index] != after.expansion_weights_q15[index]
            })
        })
        .count())
}

fn evaluate_expert(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &LowRankExpert,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    for record in records {
        let (hidden, _, _) = adapted_hidden(&record.last_hidden_q15, expert);
        let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
        let target = tokens[record.end];
        mistakes += usize::from(argmax(&row.logits_q8) != target);
        probability_error_q15 = probability_error_q15
            .saturating_add(sample_probability_error(&row.probabilities_q15, target));
    }
    Ok(Metrics {
        windows: records.len(),
        mistakes,
        probability_error_q15,
    })
}

fn teacher_probabilities(
    tokens: &[u8],
    teacher: &MiniTransformerMlpModel,
    stride: usize,
    max_windows: Option<usize>,
    student_records: &[MiniTransformerMlpWindowEvalRecord],
) -> Result<Vec<[i16; VOCAB]>, Box<dyn std::error::Error>> {
    let teacher_records = evaluate_records(tokens, teacher, stride, max_windows)?;
    if teacher_records.len() != student_records.len() {
        return Err("teacher and trunk window counts differ".into());
    }
    teacher_records
        .iter()
        .zip(student_records)
        .map(|(teacher_record, student_record)| {
            if teacher_record.start != student_record.start
                || teacher_record.end != student_record.end
            {
                return Err("teacher and trunk windows differ".into());
            }
            Ok(
                mini_transformer_output_from_hidden_q15(teacher, &teacher_record.last_hidden_q15)?
                    .probabilities_q15,
            )
        })
        .collect()
}

fn evaluate_teacher_probability_l1(
    records: &[MiniTransformerMlpWindowEvalRecord],
    model: &MiniTransformerMlpModel,
    expert: &LowRankExpert,
    teacher_probabilities: &[[i16; VOCAB]],
) -> Result<usize, Box<dyn std::error::Error>> {
    if records.len() != teacher_probabilities.len() {
        return Err("teacher probability count mismatch".into());
    }
    let mut distance = 0_usize;
    for (record, teacher) in records.iter().zip(teacher_probabilities) {
        let (hidden, _, _) = adapted_hidden(&record.last_hidden_q15, expert);
        let student = mini_transformer_output_from_hidden_q15(model, &hidden)?;
        for (&student_probability, &teacher_probability) in
            student.probabilities_q15.iter().zip(teacher)
        {
            distance = distance.saturating_add(
                (i32::from(student_probability) - i32::from(teacher_probability)).unsigned_abs()
                    as usize,
            );
        }
    }
    Ok(distance)
}

#[allow(clippy::too_many_arguments)]
fn train_expert(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut LowRankExpert,
    epochs: usize,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
    error_feedback: bool,
    teacher_probabilities: Option<&[[i16; VOCAB]]>,
) -> Result<TrainStats, Box<dyn std::error::Error>> {
    if teacher_probabilities.is_some_and(|probabilities| probabilities.len() != records.len()) {
        return Err("teacher probability count mismatch".into());
    }
    let mut residuals = vec![0_i64; expert.expansion_weights_q15.len()];
    let mut stats = TrainStats {
        optimizer_steps: 0,
        weight_delta_l1: 0,
        weight_saturation_count: 0,
        hidden_saturation_count: 0,
        residual_carry_event_count: 0,
        final_residual_carry_nonzero_count: 0,
    };
    for _ in 0..epochs {
        for (batch_number, batch) in records.chunks(batch_windows).enumerate() {
            let mut gradients = vec![0_i64; expert.expansion_weights_q15.len()];
            let batch_start = batch_number.saturating_mul(batch_windows);
            for (batch_index, record) in batch.iter().enumerate() {
                let (hidden, latent, hidden_saturations) =
                    adapted_hidden(&record.last_hidden_q15, expert);
                stats.hidden_saturation_count = stats
                    .hidden_saturation_count
                    .saturating_add(hidden_saturations);
                let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
                let mut grad_output_q15 = [0_i16; VOCAB];
                if let Some(teacher_probabilities) = teacher_probabilities {
                    let teacher = &teacher_probabilities[batch_start + batch_index];
                    for ((gradient_output, &student_probability), &teacher_probability) in
                        grad_output_q15
                            .iter_mut()
                            .zip(&row.probabilities_q15)
                            .zip(teacher)
                    {
                        *gradient_output = saturate_i16(
                            i64::from(student_probability) - i64::from(teacher_probability),
                        );
                    }
                } else {
                    let target = usize::from(tokens[record.end]);
                    let target_probability = i64::from(row.probabilities_q15[target].max(0));
                    for (index, gradient_output) in grad_output_q15.iter_mut().enumerate() {
                        let mut gradient = i64::from(row.probabilities_q15[index]);
                        if index == target {
                            gradient = gradient.saturating_sub(i64::from(i16::MAX));
                        }
                        gradient =
                            gradient.saturating_mul(target_probability) / i64::from(i16::MAX);
                        *gradient_output = saturate_i16(gradient);
                    }
                }
                let grad_hidden_q15 =
                    mini_transformer_output_gradient_to_hidden_q15(model, &grad_output_q15)?;
                for dim in 0..MINI_TRANSFORMER_D_MODEL {
                    if hidden[dim] == i16::MIN || hidden[dim] == i16::MAX {
                        continue;
                    }
                    for (rank_index, &latent_value) in latent.iter().enumerate().take(expert.rank) {
                        let index = dim * expert.rank + rank_index;
                        let gradient = round_shift_rhu_i64(
                            i64::from(grad_hidden_q15[dim]).saturating_mul(i64::from(latent_value)),
                            15,
                        );
                        gradients[index] = gradients[index].saturating_add(gradient);
                    }
                }
            }
            let denominator = i64::try_from(batch.len())?
                .checked_shl(u32::from(learning_rate_shift))
                .ok_or("low-rank denominator overflow")?;
            for index in 0..expert.expansion_weights_q15.len() {
                let numerator = gradients[index].saturating_add(if error_feedback {
                    residuals[index]
                } else {
                    0
                });
                let update = round_div_signed(numerator, denominator)?;
                residuals[index] = if error_feedback {
                    numerator.saturating_sub(update.saturating_mul(denominator))
                } else {
                    0
                };
                stats.residual_carry_event_count = stats
                    .residual_carry_event_count
                    .saturating_add(usize::from(residuals[index] != 0));
                let update = update.saturating_mul(learning_rate);
                let previous = expert.expansion_weights_q15[index];
                let raw = i64::from(previous).saturating_sub(update);
                let next = saturate_i16(raw);
                if i64::from(next) != raw {
                    stats.weight_saturation_count += 1;
                    residuals[index] = 0;
                }
                stats.weight_delta_l1 = stats
                    .weight_delta_l1
                    .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                expert.expansion_weights_q15[index] = next;
            }
            stats.optimizer_steps += 1;
        }
    }
    stats.final_residual_carry_nonzero_count =
        residuals.iter().filter(|&&residual| residual != 0).count();
    Ok(stats)
}

fn sample_probability_error(probabilities: &[i16; VOCAB], target: u8) -> usize {
    let target = usize::from(target);
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities[target])).max(0) as usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if index != target {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
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

fn route_json(value: &RouteAggregate) -> String {
    let windows = value.windows.max(1);
    let utilization = value
        .utilization_tokens
        .map(|count| count.saturating_mul(1000) / windows);
    format!(
        "{{\"windows\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{},\"decisions\":{},\"route_switches\":{},\"utilization_tokens\":[{},{},{}],\"utilization_per_mille\":[{},{},{}]}}",
        value.windows,
        value.mistakes,
        value.windows.saturating_sub(value.mistakes) * 1000 / windows,
        value.probability_error_q15,
        value.probability_error_q15 / windows,
        value.decisions,
        value.route_switches,
        value.utilization_tokens[0],
        value.utilization_tokens[1],
        value.utilization_tokens[2],
        utilization[0],
        utilization[1],
        utilization[2],
    )
}

fn decode_hex(input: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if input.is_empty() || !input.len().is_multiple_of(2) {
        return Err("prompt_hex must contain a positive even number of digits".into());
    }
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len() / 2);
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[index]).ok_or("prompt_hex contains a non-hex digit")?;
        let low = hex_nibble(bytes[index + 1]).ok_or("prompt_hex contains a non-hex digit")?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
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

fn argmax(values: &[i32; VOCAB]) -> u8 {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map_or(0, |(index, _)| index as u8)
}

fn generate_next_token(
    sequence: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &LowRankExpert,
    printable_only: bool,
    top_k: usize,
    rng_state: &mut u64,
) -> Result<u8, Box<dyn std::error::Error>> {
    let seq_len = model.context_seq_len;
    let mut window = vec![b' '; seq_len + 1];
    let context_len = sequence.len().min(seq_len);
    window[seq_len - context_len..seq_len]
        .copy_from_slice(&sequence[sequence.len() - context_len..]);
    let records = evaluate_records(&window, model, 1, Some(1))?;
    let record = records
        .first()
        .ok_or("generation produced no context window")?;
    let (hidden, _, _) = adapted_hidden(&record.last_hidden_q15, expert);
    let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
    let mut candidates = if printable_only {
        (32_u8..=126).collect::<Vec<_>>()
    } else {
        (u8::MIN..=u8::MAX).collect::<Vec<_>>()
    };
    candidates.sort_by_key(|&token| (core::cmp::Reverse(row.logits_q8[usize::from(token)]), token));
    candidates.truncate(top_k.min(candidates.len()));
    if candidates.len() == 1 {
        return Ok(candidates[0]);
    }
    let weights = candidates
        .iter()
        .map(|&token| i32::from(row.probabilities_q15[usize::from(token)]).max(1) as u64)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<u64>();
    *rng_state = rng_state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut ticket = *rng_state % total.max(1);
    for (&token, weight) in candidates.iter().zip(weights) {
        if ticket < weight {
            return Ok(token);
        }
        ticket -= weight;
    }
    Ok(*candidates.last().unwrap_or(&b' '))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn metrics_json(metrics: Metrics) -> String {
    format!(
        "{{\"windows\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{}}}",
        metrics.windows,
        metrics.mistakes,
        metrics.windows.saturating_sub(metrics.mistakes) * 1000 / metrics.windows.max(1),
        metrics.probability_error_q15,
        metrics.probability_error_q15 / metrics.windows.max(1),
    )
}

fn round_div_signed(value: i64, denominator: i64) -> Result<i64, &'static str> {
    if denominator <= 0 {
        return Err("invalid signed division denominator");
    }
    let half = denominator / 2;
    Ok(if value >= 0 {
        value.saturating_add(half) / denominator
    } else {
        value.saturating_sub(half) / denominator
    })
}

fn optional_usize_json(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn read_hidden_expert_gains(
    bytes: &[u8],
    expected_trunk_model_hash: u64,
) -> Result<[i16; MINI_TRANSFORMER_D_MODEL], Box<dyn std::error::Error>> {
    let expected_len =
        HIDDEN_EXPERT_MAGIC.len() + 8 + MINI_TRANSFORMER_D_MODEL * size_of::<i16>() + 8;
    if bytes.len() != expected_len || &bytes[..HIDDEN_EXPERT_MAGIC.len()] != HIDDEN_EXPERT_MAGIC {
        return Err("invalid base hidden expert artifact".into());
    }
    let checksum_offset = bytes.len() - 8;
    let expected_checksum = u64::from_le_bytes(bytes[checksum_offset..].try_into()?);
    if fnv64(&bytes[..checksum_offset]) != expected_checksum {
        return Err("base hidden expert checksum mismatch".into());
    }
    let mut offset = HIDDEN_EXPERT_MAGIC.len();
    let trunk_model_hash = read_u64(bytes, &mut offset)?;
    if trunk_model_hash != expected_trunk_model_hash {
        return Err("base hidden expert is bound to a different trunk".into());
    }
    let mut gains = [0_i16; MINI_TRANSFORMER_D_MODEL];
    for gain in &mut gains {
        *gain = read_i16(bytes, &mut offset)?;
    }
    Ok(gains)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = offset.checked_add(4).ok_or("offset overflow")?;
    let value = u32::from_le_bytes(bytes.get(*offset..end).ok_or("truncated u32")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, Box<dyn std::error::Error>> {
    let end = offset.checked_add(8).ok_or("offset overflow")?;
    let value = u64::from_le_bytes(bytes.get(*offset..end).ok_or("truncated u64")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn read_i16(bytes: &[u8], offset: &mut usize) -> Result<i16, Box<dyn std::error::Error>> {
    let end = offset.checked_add(2).ok_or("offset overflow")?;
    let value = i16::from_le_bytes(bytes.get(*offset..end).ok_or("truncated i16")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_expansion_is_identity() {
        let expert =
            LowRankExpert::new_with_base(1, 4, 7, [0; MINI_TRANSFORMER_D_MODEL]).expect("expert");
        let hidden = core::array::from_fn(|index| index as i16 - 64);
        let (adapted, latent, saturations) = adapted_hidden(&hidden, &expert);
        assert_eq!(adapted, hidden);
        assert_eq!(latent.len(), 4);
        assert_eq!(saturations, 0);
    }

    #[test]
    fn zero_expansion_preserves_diagonal_base() {
        let mut gains = [0_i16; MINI_TRANSFORMER_D_MODEL];
        gains[0] = 16_384;
        gains[1] = -16_384;
        let expert = LowRankExpert::new_with_base(1, 4, 7, gains).expect("expert");
        let mut hidden = [0_i16; MINI_TRANSFORMER_D_MODEL];
        hidden[0] = 1_000;
        hidden[1] = 1_000;
        let (adapted, _, saturations) = adapted_hidden(&hidden, &expert);
        assert_eq!(adapted[0], 1_500);
        assert_eq!(adapted[1], 500);
        assert_eq!(saturations, 0);
    }

    #[test]
    fn artifact_round_trip_and_corruption_are_locked() {
        let mut base_gains = [0_i16; MINI_TRANSFORMER_D_MODEL];
        base_gains[3] = 123;
        let mut expert = LowRankExpert::new_with_base(9, 4, 11, base_gains).expect("expert");
        expert.expansion_weights_q15[5] = -321;
        let bytes = expert.to_bytes().expect("bytes");
        assert_eq!(LowRankExpert::from_bytes(&bytes).expect("decode"), expert);
        let mut corrupt = bytes;
        corrupt[30] ^= 1;
        assert!(LowRankExpert::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn update_fingerprint_records_exact_delta_geometry() {
        let before = [0_i16, -2, 7, i16::MAX];
        let after = [0_i16, 3, 4, i16::MIN];
        let fingerprint = update_fingerprint(&before, &after).expect("fingerprint");
        assert_eq!(fingerprint.element_count, 4);
        assert_eq!(fingerprint.nonzero_count, 3);
        assert_eq!(fingerprint.delta_l1, 65_543);
        assert_eq!(fingerprint.delta_linf, 65_535);
        assert_ne!(
            fingerprint.hash,
            update_fingerprint(&before, &before).expect("identity").hash
        );
    }

    #[test]
    fn active_update_rank_counts_only_changed_columns() {
        let before =
            LowRankExpert::new_with_base(1, 4, 7, [0; MINI_TRANSFORMER_D_MODEL]).expect("expert");
        let mut after = before.clone();
        after.expansion_weights_q15[1] = 1;
        after.expansion_weights_q15[MINI_TRANSFORMER_D_MODEL + 3] = -1;
        assert_eq!(active_update_rank(&before, &after).expect("rank"), 2);
    }

    #[test]
    fn version_one_artifacts_decode_with_zero_base() {
        let expert =
            LowRankExpert::new_with_base(9, 4, 11, [0; MINI_TRANSFORMER_D_MODEL]).expect("expert");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V1);
        bytes.extend_from_slice(&expert.trunk_model_hash.to_le_bytes());
        bytes.extend_from_slice(&(expert.rank as u32).to_le_bytes());
        bytes.extend_from_slice(&expert.projection_seed.to_le_bytes());
        for &value in &expert.expansion_weights_q15 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = fnv64(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(LowRankExpert::from_bytes(&bytes).expect("decode"), expert);
    }

    #[test]
    fn identical_teacher_has_zero_distillation_update() {
        let tokens = b"To be or not to be";
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let records = evaluate_records(tokens, &model, 1, Some(4)).expect("records");
        let teacher = teacher_probabilities(tokens, &model, 1, Some(4), &records)
            .expect("teacher probabilities");
        let mut expert = LowRankExpert::new_with_base(
            model.model_hash(),
            4,
            DEFAULT_PROJECTION_SEED,
            [0; MINI_TRANSFORMER_D_MODEL],
        )
        .expect("expert");
        let stats = train_expert(
            &records,
            tokens,
            &model,
            &mut expert,
            1,
            records.len(),
            16_384,
            0,
            true,
            Some(&teacher),
        )
        .expect("distill");
        assert_eq!(stats.weight_delta_l1, 0);
        assert!(
            expert
                .expansion_weights_q15
                .iter()
                .all(|&weight| weight == 0)
        );
        assert_eq!(
            evaluate_teacher_probability_l1(&records, &model, &expert, &teacher).expect("distance"),
            0
        );
    }

    #[test]
    fn adapter_generation_is_deterministic_and_printable() {
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        let expert = LowRankExpert::new_with_base(
            model.model_hash(),
            4,
            DEFAULT_PROJECTION_SEED,
            [0; MINI_TRANSFORMER_D_MODEL],
        )
        .expect("expert");
        let mut left_rng = 7;
        let mut right_rng = 7;
        let left = generate_next_token(b"the soul", &model, &expert, true, 8, &mut left_rng)
            .expect("token");
        let right = generate_next_token(b"the soul", &model, &expert, true, 8, &mut right_rng)
            .expect("token");
        assert_eq!(left, right);
        assert!((32..=126).contains(&left));
    }
}
