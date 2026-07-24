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

const MAGIC: &[u8; 8] = b"NSRLHE1\n";
const VOCAB: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
struct HiddenExpert {
    trunk_model_hash: u64,
    gains_q15: [i16; MINI_TRANSFORMER_D_MODEL],
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
    gain_delta_l1: u64,
    gain_saturation_count: usize,
    hidden_saturation_count: usize,
}

#[derive(Clone, Copy)]
struct HiddenTrainConfig {
    epochs: usize,
    batch_windows: usize,
    learning_rate_shift: u8,
    learning_rate: i64,
    reverse_gradient: bool,
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
        eprintln!("nsrl-mini-transformer-hidden-expert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => run_train(args),
        Some("eval") => run_eval(args),
        Some("score") => run_score(args),
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        _ => Err("expected train, eval, or score".into()),
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
    let mut learning_rate_shift = 0_u8;
    let mut learning_rate = 1_i64;
    let mut reverse_gradient = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => expert_out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--epochs" => epochs = required(&mut args, "--epochs")?.parse()?,
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--batch-windows" => batch_windows = required(&mut args, "--batch-windows")?.parse()?,
            "--learning-rate-shift" => {
                learning_rate_shift = required(&mut args, "--learning-rate-shift")?.parse()?
            }
            "--learning-rate" => learning_rate = required(&mut args, "--learning-rate")?.parse()?,
            "--reverse-gradient" => reverse_gradient = true,
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    if epochs == 0
        || stride == 0
        || batch_windows == 0
        || learning_rate_shift > 30
        || !(1..=8192).contains(&learning_rate)
    {
        return Err("invalid training configuration".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let mut expert = HiddenExpert {
        trunk_model_hash: model.model_hash(),
        gains_q15: [0; MINI_TRANSFORMER_D_MODEL],
    };
    let initial = evaluate_hidden(&records, &tokens, &model, &expert.gains_q15)?;
    let stats = train_hidden(
        &records,
        &tokens,
        &model,
        &mut expert.gains_q15,
        HiddenTrainConfig {
            epochs,
            batch_windows,
            learning_rate_shift,
            learning_rate,
            reverse_gradient,
        },
    )?;
    let final_metrics = evaluate_hidden(&records, &tokens, &model, &expert.gains_q15)?;
    let artifact = expert.to_bytes();
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_hidden_expert_train.v1\",\"expert_type\":\"diagonal_hidden_residual_q15\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"config\":{{\"epochs\":{},\"stride\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate\":{},\"learning_rate_shift\":{},\"objective\":\"probability_error\",\"reverse_gradient_diagnostic\":{}}},\"initial\":{},\"final\":{},\"updates\":{{\"optimizer_steps\":{},\"gain_delta_l1\":{},\"gain_saturation_count\":{},\"hidden_saturation_count\":{}}}}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            epochs,
            stride,
            optional_usize_json(max_windows),
            batch_windows.min(records.len()),
            learning_rate,
            learning_rate_shift,
            reverse_gradient,
            metrics_json(initial),
            metrics_json(final_metrics),
            stats.optimizer_steps,
            stats.gain_delta_l1,
            stats.gain_saturation_count,
            stats.hidden_saturation_count,
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
    let expert = HiddenExpert::from_bytes(&expert_bytes)?;
    if expert.trunk_model_hash != model.model_hash() {
        return Err("hidden expert is bound to a different trunk".into());
    }
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let metrics = evaluate_hidden(&records, &tokens, &model, &expert.gains_q15)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_hidden_expert_eval.v1\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"stride\":{},\"max_windows\":{},\"metrics\":{}}}\n",
            model.model_hash(),
            fnv64(&expert_bytes),
            stride,
            optional_usize_json(max_windows),
            metrics_json(metrics),
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
        let expert = HiddenExpert::from_bytes(&bytes)?;
        if expert.trunk_model_hash != model.model_hash() {
            return Err("hidden expert is bound to a different trunk".into());
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
                let (hidden, _) = adapted_hidden(&record.last_hidden_q15, &expert.gains_q15);
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
        "{{\"schema\":\"nsrl.shared_trunk_hidden_routing_ablation.v1\",\"dataset\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"windows\":{},\"stride\":{}}},\"trunk\":{{\"model_hash\":\"0x{:016x}\",\"forward_count\":{},\"naive_three_model_forward_count\":{}}},\"experts\":{{\"ids\":[{},{},{}],\"artifact_hashes\":[\"0x{:016x}\",\"0x{:016x}\",\"0x{:016x}\"],\"parameter_count_each\":{}}},\"fixed_experts\":[{},{},{}],\"best_fixed_expert\":{},\"oracle_routes\":{{\"prompt\":{},\"span\":{},\"token\":{}}},\"known_non_claims\":[\"target_aware_oracle_ceiling\",\"diagonal_hidden_experts_not_low_rank_ffn_experts_yet\",\"does_not_claim_language_model_quality\"]}}",
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
        MINI_TRANSFORMER_D_MODEL,
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
        "  nsrl-mini-transformer-hidden-expert train --tokens PATH --model PATH --out PATH --trace PATH [--epochs N] [--stride N] [--max-windows N] [--batch-windows N] [--learning-rate N] [--learning-rate-shift N] [--reverse-gradient]"
    );
    println!(
        "  nsrl-mini-transformer-hidden-expert eval --tokens PATH --model PATH --expert PATH --trace PATH [--stride N] [--max-windows N]"
    );
    println!(
        "  nsrl-mini-transformer-hidden-expert score --input PATH --model PATH --expert ID=PATH --expert ID=PATH --expert ID=PATH --out PATH --details-out PATH [--stride N] [--span-len N] [--max-samples N]"
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

fn adapted_hidden(
    base: &[i16; MINI_TRANSFORMER_D_MODEL],
    gains_q15: &[i16; MINI_TRANSFORMER_D_MODEL],
) -> ([i16; MINI_TRANSFORMER_D_MODEL], usize) {
    let mut output = [0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut saturation_count = 0_usize;
    for index in 0..MINI_TRANSFORMER_D_MODEL {
        let residual = round_shift_rhu_i64(
            i64::from(base[index]).saturating_mul(i64::from(gains_q15[index])),
            15,
        );
        let raw = i64::from(base[index]).saturating_add(residual);
        output[index] = saturate_i16(raw);
        saturation_count += usize::from(i64::from(output[index]) != raw);
    }
    (output, saturation_count)
}

fn evaluate_hidden(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    gains_q15: &[i16; MINI_TRANSFORMER_D_MODEL],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    for record in records {
        let (hidden, _) = adapted_hidden(&record.last_hidden_q15, gains_q15);
        let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
        let target = *tokens.get(record.end).ok_or("target is out of range")?;
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

fn train_hidden(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    gains_q15: &mut [i16; MINI_TRANSFORMER_D_MODEL],
    config: HiddenTrainConfig,
) -> Result<TrainStats, Box<dyn std::error::Error>> {
    let mut residuals = [0_i64; MINI_TRANSFORMER_D_MODEL];
    let mut stats = TrainStats {
        optimizer_steps: 0,
        gain_delta_l1: 0,
        gain_saturation_count: 0,
        hidden_saturation_count: 0,
    };
    for _ in 0..config.epochs {
        for batch in records.chunks(config.batch_windows) {
            let mut gradients = [0_i64; MINI_TRANSFORMER_D_MODEL];
            for record in batch {
                let (hidden, hidden_saturations) =
                    adapted_hidden(&record.last_hidden_q15, gains_q15);
                stats.hidden_saturation_count = stats
                    .hidden_saturation_count
                    .saturating_add(hidden_saturations);
                let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
                let target = usize::from(*tokens.get(record.end).ok_or("target out of range")?);
                let target_probability = i64::from(row.probabilities_q15[target].max(0));
                let mut grad_output_q15 = [0_i16; VOCAB];
                for (index, gradient_output) in grad_output_q15.iter_mut().enumerate() {
                    let mut gradient = i64::from(row.probabilities_q15[index]);
                    if index == target {
                        gradient = gradient.saturating_sub(i64::from(i16::MAX));
                    }
                    gradient = gradient.saturating_mul(target_probability) / i64::from(i16::MAX);
                    *gradient_output = saturate_i16(gradient);
                }
                let grad_hidden_q15 =
                    mini_transformer_output_gradient_to_hidden_q15(model, &grad_output_q15)?;
                for index in 0..MINI_TRANSFORMER_D_MODEL {
                    if hidden[index] == i16::MIN || hidden[index] == i16::MAX {
                        continue;
                    }
                    let gradient = round_shift_rhu_i64(
                        i64::from(grad_hidden_q15[index])
                            .saturating_mul(i64::from(record.last_hidden_q15[index])),
                        15,
                    );
                    gradients[index] = gradients[index].saturating_add(gradient);
                }
            }
            let denominator = i64::try_from(batch.len())?
                .checked_shl(u32::from(config.learning_rate_shift))
                .ok_or("hidden expert denominator overflow")?;
            for index in 0..MINI_TRANSFORMER_D_MODEL {
                let numerator = gradients[index].saturating_add(residuals[index]);
                let update = round_div_signed(numerator, denominator)?;
                residuals[index] = numerator.saturating_sub(update.saturating_mul(denominator));
                let update = update.saturating_mul(config.learning_rate);
                let previous = gains_q15[index];
                let raw = if config.reverse_gradient {
                    i64::from(previous).saturating_add(update)
                } else {
                    i64::from(previous).saturating_sub(update)
                };
                let next = saturate_i16(raw);
                if i64::from(next) != raw {
                    stats.gain_saturation_count += 1;
                    residuals[index] = 0;
                }
                stats.gain_delta_l1 = stats
                    .gain_delta_l1
                    .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                gains_q15[index] = next;
            }
            stats.optimizer_steps += 1;
        }
    }
    Ok(stats)
}

impl HiddenExpert {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MAGIC.len() + 8 + MINI_TRANSFORMER_D_MODEL * 2 + 8);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        for value in self.gains_q15 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = fnv64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let expected_len = MAGIC.len() + 8 + MINI_TRANSFORMER_D_MODEL * 2 + 8;
        if bytes.len() != expected_len || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid hidden expert artifact".into());
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(bytes[checksum_offset..].try_into()?);
        if fnv64(&bytes[..checksum_offset]) != expected_checksum {
            return Err("hidden expert checksum mismatch".into());
        }
        let mut offset = MAGIC.len();
        let trunk_model_hash = read_u64(bytes, &mut offset)?;
        let mut gains_q15 = [0_i16; MINI_TRANSFORMER_D_MODEL];
        for value in &mut gains_q15 {
            *value = read_i16(bytes, &mut offset)?;
        }
        Ok(Self {
            trunk_model_hash,
            gains_q15,
        })
    }
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
    fn zero_gain_is_identity() {
        let hidden = core::array::from_fn(|index| index as i16 - 64);
        assert_eq!(
            adapted_hidden(&hidden, &[0; MINI_TRANSFORMER_D_MODEL]),
            (hidden, 0)
        );
    }

    #[test]
    fn artifact_round_trip_and_corruption_are_locked() {
        let mut expert = HiddenExpert {
            trunk_model_hash: 99,
            gains_q15: [0; MINI_TRANSFORMER_D_MODEL],
        };
        expert.gains_q15[3] = -100;
        let bytes = expert.to_bytes();
        assert_eq!(HiddenExpert::from_bytes(&bytes).expect("decode"), expert);
        let mut corrupt = bytes;
        corrupt[24] ^= 1;
        assert!(HiddenExpert::from_bytes(&corrupt).is_err());
    }
}
