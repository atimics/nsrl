#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_core::base2_softmax_i32_q15;
use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerMlpWindowEvalRecord, MiniTransformerPositionPolicy,
    evaluate_mini_transformer_mlp_windows,
};

const MAGIC: &[u8; 8] = b"NSRLBE1\n";
const VOCAB: usize = 256;
const BIAS_LIMIT_Q8: i32 = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BiasObjective {
    CrossEntropy,
    ProbabilityError,
}

impl BiasObjective {
    fn as_str(self) -> &'static str {
        match self {
            Self::CrossEntropy => "cross_entropy",
            Self::ProbabilityError => "probability_error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BiasExpert {
    trunk_model_hash: u64,
    biases_q8: [i32; VOCAB],
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
    bias_delta_l1: u64,
    clipped_update_count: usize,
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
        eprintln!("nsrl-mini-transformer-bias-expert: {error}");
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
    let mut batch_windows = 256_usize;
    let mut learning_rate_shift = 8_u8;
    let mut objective = BiasObjective::ProbabilityError;
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
            "--objective" => {
                objective = match required(&mut args, "--objective")?.as_str() {
                    "cross-entropy" => BiasObjective::CrossEntropy,
                    "probability-error" => BiasObjective::ProbabilityError,
                    _ => {
                        return Err(
                            "--objective requires cross-entropy or probability-error".into()
                        );
                    }
                }
            }
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    if epochs == 0 || stride == 0 || batch_windows == 0 || learning_rate_shift > 30 {
        return Err("epochs, stride, and batch size must be positive; shift must be <= 30".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let mut expert = BiasExpert {
        trunk_model_hash: model.model_hash(),
        biases_q8: [0; VOCAB],
    };
    let initial = evaluate_bias(&records, &tokens, &expert.biases_q8)?;
    let stats = train_bias(
        &records,
        &tokens,
        &mut expert.biases_q8,
        epochs,
        batch_windows,
        learning_rate_shift,
        objective,
    )?;
    let final_metrics = evaluate_bias(&records, &tokens, &expert.biases_q8)?;
    let artifact = expert.to_bytes();
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_bias_expert_train.v1\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"config\":{{\"epochs\":{},\"stride\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate_shift\":{},\"objective\":\"{}\"}},\"initial\":{},\"final\":{},\"updates\":{{\"optimizer_steps\":{},\"bias_delta_l1\":{},\"clipped_update_count\":{}}}}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            epochs,
            stride,
            optional_usize_json(max_windows),
            batch_windows,
            learning_rate_shift,
            objective.as_str(),
            metrics_json(initial),
            metrics_json(final_metrics),
            stats.optimizer_steps,
            stats.bias_delta_l1,
            stats.clipped_update_count,
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
    let expert = BiasExpert::from_bytes(&expert_bytes)?;
    if expert.trunk_model_hash != model.model_hash() {
        return Err("bias expert is bound to a different trunk".into());
    }
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let metrics = evaluate_bias(&records, &tokens, &expert.biases_q8)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_bias_expert_eval.v1\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"stride\":{},\"max_windows\":{},\"metrics\":{}}}\n",
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
        return Err("score requires exactly three experts and positive stride/span".into());
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
        let expert = BiasExpert::from_bytes(&bytes)?;
        if expert.trunk_model_hash != model.model_hash() {
            return Err("bias expert is bound to a different trunk".into());
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
            let target = *tokens.get(record.end).ok_or("score target out of range")?;
            for expert_index in 0..3 {
                let (logits, probabilities) = biased_row(record, &experts[expert_index].biases_q8)?;
                losses[index][expert_index] = sample_probability_error(&probabilities, target);
                mistakes[index][expert_index] = usize::from(argmax(&logits) != target);
            }
        }
        for expert in 0..3 {
            fixed[expert].add_group(&losses, &mistakes, 0, records.len(), expert);
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
        for index in 0..records.len() {
            let choice = best_expert(&losses, &mistakes, index, index + 1);
            if previous_token.is_some_and(|previous| previous != choice) {
                token.route_switches = token.route_switches.saturating_add(1);
            }
            previous_token = Some(choice);
            token.add_group(&losses, &mistakes, index, index + 1, choice);
            token_routes[index] = choice;
        }

        for (index, record) in records.iter().enumerate() {
            let target = tokens[record.end];
            writeln!(
                details,
                "{}\t{}\t{:02x}\t{},{},{}\t{},{},{}\t{}\t{}\t{}\t{}",
                sample_id,
                record.end,
                target,
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
        "{{\"schema\":\"nsrl.shared_trunk_bias_routing_ablation.v1\",\"dataset\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"windows\":{},\"stride\":{}}},\"trunk\":{{\"model_hash\":\"0x{:016x}\",\"forward_count\":{},\"naive_three_model_forward_count\":{}}},\"experts\":{{\"ids\":[{},{},{}],\"artifact_hashes\":[\"0x{:016x}\",\"0x{:016x}\",\"0x{:016x}\"],\"parameter_count_each\":256}},\"fixed_experts\":[{},{},{}],\"best_fixed_expert\":{},\"oracle_routes\":{{\"prompt\":{},\"span\":{},\"token\":{}}},\"known_non_claims\":[\"target_aware_oracle_ceiling\",\"bias_experts_not_ffn_experts_yet\",\"does_not_claim_language_model_quality\"]}}",
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
        "  nsrl-mini-transformer-bias-expert train --tokens PATH --model PATH --out PATH --trace PATH [--epochs N] [--stride N] [--max-windows N] [--batch-windows N] [--learning-rate-shift N] [--objective cross-entropy|probability-error]"
    );
    println!(
        "  nsrl-mini-transformer-bias-expert eval --tokens PATH --model PATH --expert PATH --trace PATH [--stride N] [--max-windows N]"
    );
    println!(
        "  nsrl-mini-transformer-bias-expert score --input PATH --model PATH --expert ID=PATH --expert ID=PATH --expert ID=PATH --out PATH --details-out PATH [--stride N] [--span-len N] [--max-samples N]"
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

fn biased_row(
    record: &MiniTransformerMlpWindowEvalRecord,
    biases_q8: &[i32; VOCAB],
) -> Result<([i32; VOCAB], [i16; VOCAB]), Box<dyn std::error::Error>> {
    let base = record
        .logits_q8
        .as_ref()
        .ok_or("trunk forward was invalid")?;
    let logits = core::array::from_fn(|index| base[index].saturating_add(biases_q8[index]));
    let mut probabilities = [0_i16; VOCAB];
    base2_softmax_i32_q15(&logits, &mut probabilities).ok_or("bias softmax failed")?;
    Ok((logits, probabilities))
}

fn evaluate_bias(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    biases_q8: &[i32; VOCAB],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    for record in records {
        let (logits, probabilities) = biased_row(record, biases_q8)?;
        let target = *tokens.get(record.end).ok_or("target is out of range")?;
        mistakes += usize::from(argmax(&logits) != target);
        probability_error_q15 =
            probability_error_q15.saturating_add(sample_probability_error(&probabilities, target));
    }
    Ok(Metrics {
        windows: records.len(),
        mistakes,
        probability_error_q15,
    })
}

fn train_bias(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    biases_q8: &mut [i32; VOCAB],
    epochs: usize,
    batch_windows: usize,
    learning_rate_shift: u8,
    objective: BiasObjective,
) -> Result<TrainStats, Box<dyn std::error::Error>> {
    let mut residuals = [0_i64; VOCAB];
    let mut stats = TrainStats {
        optimizer_steps: 0,
        bias_delta_l1: 0,
        clipped_update_count: 0,
    };
    for _ in 0..epochs {
        for batch in records.chunks(batch_windows) {
            let mut gradients = [0_i64; VOCAB];
            for record in batch {
                let (_, probabilities) = biased_row(record, biases_q8)?;
                let target = usize::from(*tokens.get(record.end).ok_or("target out of range")?);
                let target_probability = i64::from(probabilities[target].max(0));
                for index in 0..VOCAB {
                    let mut gradient = i64::from(probabilities[index]);
                    if index == target {
                        gradient = gradient.saturating_sub(i64::from(i16::MAX));
                    }
                    if objective == BiasObjective::ProbabilityError {
                        gradient =
                            gradient.saturating_mul(target_probability) / i64::from(i16::MAX);
                    }
                    gradients[index] = gradients[index]
                        .checked_add(gradient)
                        .ok_or("bias gradient overflow")?;
                }
            }
            let denominator = i64::try_from(batch.len())?
                .checked_shl(u32::from(learning_rate_shift))
                .ok_or("bias denominator overflow")?;
            for index in 0..VOCAB {
                let numerator = gradients[index]
                    .checked_add(residuals[index])
                    .ok_or("bias residual overflow")?;
                let update = round_div_signed(numerator, denominator)?;
                residuals[index] = numerator.saturating_sub(update.saturating_mul(denominator));
                let previous = biases_q8[index];
                let unclipped = i64::from(previous).saturating_sub(update);
                let next =
                    unclipped.clamp(-i64::from(BIAS_LIMIT_Q8), i64::from(BIAS_LIMIT_Q8)) as i32;
                if i64::from(next) != unclipped {
                    stats.clipped_update_count += 1;
                    residuals[index] = 0;
                }
                stats.bias_delta_l1 = stats
                    .bias_delta_l1
                    .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                biases_q8[index] = next;
            }
            stats.optimizer_steps += 1;
        }
    }
    Ok(stats)
}

impl BiasExpert {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 8 + VOCAB * 4 + 8);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        for value in self.biases_q8 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = fnv64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let expected_len = MAGIC.len() + 8 + VOCAB * 4 + 8;
        if bytes.len() != expected_len || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid bias expert artifact".into());
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(bytes[checksum_offset..].try_into()?);
        if fnv64(&bytes[..checksum_offset]) != expected_checksum {
            return Err("bias expert checksum mismatch".into());
        }
        let mut offset = MAGIC.len();
        let trunk_model_hash = read_u64(bytes, &mut offset)?;
        let mut biases_q8 = [0_i32; VOCAB];
        for value in &mut biases_q8 {
            *value = read_i32(bytes, &mut offset)?;
        }
        Ok(Self {
            trunk_model_hash,
            biases_q8,
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

fn read_i32(bytes: &[u8], offset: &mut usize) -> Result<i32, Box<dyn std::error::Error>> {
    let end = offset.checked_add(4).ok_or("offset overflow")?;
    let value = i32::from_le_bytes(bytes.get(*offset..end).ok_or("truncated i32")?.try_into()?);
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
    fn artifact_round_trip_and_checksum_are_locked() {
        let mut expert = BiasExpert {
            trunk_model_hash: 7,
            biases_q8: [0; VOCAB],
        };
        expert.biases_q8[10] = -123;
        expert.biases_q8[200] = 456;
        let bytes = expert.to_bytes();
        assert_eq!(BiasExpert::from_bytes(&bytes).expect("decode"), expert);
        let mut corrupt = bytes;
        corrupt[20] ^= 1;
        assert!(BiasExpert::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn signed_rounding_is_symmetric() {
        assert_eq!(round_div_signed(6, 4).expect("positive"), 2);
        assert_eq!(round_div_signed(-6, 4).expect("negative"), -2);
        assert_eq!(round_div_signed(1, 4).expect("small"), 0);
        assert_eq!(round_div_signed(-1, 4).expect("small negative"), 0);
    }
}
