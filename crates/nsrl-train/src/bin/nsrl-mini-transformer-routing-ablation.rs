#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerMlpWindowEvalRecord, MiniTransformerPositionPolicy, TrainError,
    evaluate_mini_transformer_mlp_windows,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Aggregate {
    windows: usize,
    mistakes: usize,
    probability_error_q15: usize,
    invalid_forward_count: usize,
    decisions: usize,
    route_switches: usize,
    utilization_tokens: [usize; 3],
}

impl Aggregate {
    fn add_group(&mut self, records: &[MiniTransformerMlpWindowEvalRecord], expert: usize) {
        self.decisions = self.decisions.saturating_add(1);
        self.utilization_tokens[expert] =
            self.utilization_tokens[expert].saturating_add(records.len());
        for record in records {
            self.windows = self.windows.saturating_add(1);
            self.mistakes = self.mistakes.saturating_add(record.mistakes);
            self.probability_error_q15 = self
                .probability_error_q15
                .saturating_add(record.probability_error_q15);
            self.invalid_forward_count = self
                .invalid_forward_count
                .saturating_add(record.invalid_forward_count);
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-routing-ablation: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input_path = None;
    let mut output_path = None;
    let mut details_path = None;
    let mut model_specs = Vec::new();
    let mut stride = 1_usize;
    let mut span_len = 16_usize;
    let mut max_samples = None;
    let mut router_feature_model_index = 2_usize;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input_path = Some(PathBuf::from(required(&mut args, "--input")?)),
            "--out" => output_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--details-out" => {
                details_path = Some(PathBuf::from(required(&mut args, "--details-out")?))
            }
            "--model" => model_specs.push(required(&mut args, "--model")?),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--span-len" => span_len = required(&mut args, "--span-len")?.parse()?,
            "--max-samples" => max_samples = Some(required(&mut args, "--max-samples")?.parse()?),
            "--router-feature-model-index" => {
                router_feature_model_index =
                    required(&mut args, "--router-feature-model-index")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mini-transformer-routing-ablation --input PATH --out PATH \
                     --model ID=PATH --model ID=PATH --model ID=PATH \
                     [--details-out PATH] [--span-len N] [--stride N] [--max-samples N] \
                     [--router-feature-model-index 0|1|2]\n\
                     Input TSV: sample_id<TAB>prompt_hex"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if model_specs.len() != 3 {
        return Err("exactly three --model ID=PATH arguments are required".into());
    }
    if stride == 0 || span_len == 0 || max_samples == Some(0) {
        return Err("stride, span length, and max samples must be positive".into());
    }
    if router_feature_model_index >= 3 {
        return Err("--router-feature-model-index must be 0, 1, or 2".into());
    }
    let input_path = input_path.ok_or("--input is required")?;
    let output_path = output_path.ok_or("--out is required")?;

    let mut model_ids = Vec::with_capacity(3);
    let mut models = Vec::with_capacity(3);
    for spec in model_specs {
        let (id, path) = spec.split_once('=').ok_or("--model requires ID=PATH")?;
        if id.is_empty() || path.is_empty() || id.contains(['\t', ',', '=']) {
            return Err("model ID/path is invalid".into());
        }
        model_ids.push(id.to_string());
        models.push(MiniTransformerMlpModel::from_bytes(&fs::read(path)?)?);
    }
    let seq_len = models[0].context_seq_len;
    if models.iter().any(|model| model.context_seq_len != seq_len) {
        return Err("all models must use the same context length".into());
    }

    let input_bytes = fs::read(&input_path)?;
    let input_hash = stable_hash(&input_bytes);
    let input = BufReader::new(input_bytes.as_slice());
    let mut details = details_path
        .map(fs::File::create)
        .transpose()?
        .map(BufWriter::new);
    if let Some(output) = details.as_mut() {
        writeln!(
            output,
            "sample_id\ttarget_offset\ttarget_hex\tchild_probability_error_q15\tchild_mistakes\trouter_hidden_features_q15\tprompt_route\tspan_route\ttoken_route"
        )?;
    }

    let mut fixed = [
        Aggregate::default(),
        Aggregate::default(),
        Aggregate::default(),
    ];
    let mut prompt = Aggregate::default();
    let mut span = Aggregate::default();
    let mut token = Aggregate::default();
    let mut sample_count = 0_usize;

    for (line_index, line) in input.lines().enumerate() {
        let line = line?;
        if line_index == 0 && line == "sample_id\tprompt_hex" {
            continue;
        }
        if max_samples.is_some_and(|limit| sample_count >= limit) {
            break;
        }
        let (sample_id, prompt_hex) = line
            .split_once('\t')
            .ok_or("input row requires sample_id and prompt_hex")?;
        let bytes = decode_hex(prompt_hex)?;
        if bytes.len() <= seq_len {
            return Err(format!("sample is too short: {sample_id}").into());
        }
        let config = MiniTransformerMlpEvalConfig {
            seq_len,
            stride,
            max_windows: None,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
        };
        let records = std::thread::scope(|scope| {
            let handles = models
                .iter()
                .map(|model| {
                    scope.spawn(|| evaluate_mini_transformer_mlp_windows(&bytes, model, config))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| TrainError::InvalidModel("routing ablation worker panicked"))?
                })
                .collect::<Result<Vec<_>, TrainError>>()
        })?;
        let windows = records[0].len();
        if windows == 0
            || records.iter().any(|rows| rows.len() != windows)
            || (0..windows).any(|index| {
                records
                    .iter()
                    .any(|rows| rows[index].start != records[0][index].start)
            })
        {
            return Err("model window records do not align".into());
        }

        for (expert_index, aggregate) in fixed.iter_mut().enumerate() {
            aggregate.add_group(&records[expert_index], expert_index);
        }

        let prompt_choice = best_expert(&records, 0, windows);
        prompt.add_group(&records[prompt_choice], prompt_choice);

        let mut span_routes = vec![0_usize; windows];
        let mut previous_span = None;
        for start in (0..windows).step_by(span_len) {
            let end = (start + span_len).min(windows);
            let choice = best_expert(&records, start, end);
            if previous_span.is_some_and(|previous| previous != choice) {
                span.route_switches = span.route_switches.saturating_add(1);
            }
            previous_span = Some(choice);
            span.add_group(&records[choice][start..end], choice);
            span_routes[start..end].fill(choice);
        }

        let mut token_routes = vec![0_usize; windows];
        let mut previous_token = None;
        for (index, route) in token_routes.iter_mut().enumerate() {
            let choice = best_expert(&records, index, index + 1);
            if previous_token.is_some_and(|previous| previous != choice) {
                token.route_switches = token.route_switches.saturating_add(1);
            }
            previous_token = Some(choice);
            token.add_group(&records[choice][index..index + 1], choice);
            *route = choice;
        }

        if let Some(output) = details.as_mut() {
            for index in 0..windows {
                let end = records[0][index].end;
                writeln!(
                    output,
                    "{}\t{}\t{:02x}\t{},{},{}\t{},{},{}\t{}\t{}\t{}\t{}",
                    sample_id,
                    end,
                    bytes[end],
                    records[0][index].probability_error_q15,
                    records[1][index].probability_error_q15,
                    records[2][index].probability_error_q15,
                    records[0][index].mistakes,
                    records[1][index].mistakes,
                    records[2][index].mistakes,
                    records[router_feature_model_index][index]
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
        }
        sample_count += 1;
    }
    if sample_count == 0 {
        return Err("input contains no scoring rows".into());
    }
    if let Some(output) = details.as_mut() {
        output.flush()?;
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
    let mut output = BufWriter::new(fs::File::create(output_path)?);
    writeln!(
        output,
        "{{\"schema\":\"nsrl.mini_transformer.routing_granularity_ablation.v2\",\"dataset\":{{\"path\":{},\"hash\":\"0x{:016x}\",\"samples\":{},\"stride\":{},\"windows\":{}}},\"models\":{{\"ids\":[{},{},{}],\"hashes\":[\"0x{:016x}\",\"0x{:016x}\",\"0x{:016x}\"],\"seq_len\":{},\"router_feature_model_index\":{}}},\"fixed_experts\":[{},{},{}],\"best_fixed_expert\":{},\"oracle_routes\":{{\"prompt\":{},\"span\":{},\"token\":{}}},\"known_non_claims\":[\"target-aware_oracle_ceiling_not_deployable_router\",\"whole_model_experts_not_shared_attention_moe\",\"does_not_claim_language_model_quality\"]}}",
        json_string(&input_path.to_string_lossy()),
        input_hash,
        sample_count,
        stride,
        prompt.windows,
        json_string(&model_ids[0]),
        json_string(&model_ids[1]),
        json_string(&model_ids[2]),
        models[0].model_hash(),
        models[1].model_hash(),
        models[2].model_hash(),
        seq_len,
        router_feature_model_index,
        aggregate_json(&fixed[0]),
        aggregate_json(&fixed[1]),
        aggregate_json(&fixed[2]),
        best_fixed,
        aggregate_json(&prompt),
        aggregate_json(&span),
        aggregate_json(&token),
    )?;
    output.flush()?;
    Ok(())
}

fn best_expert(
    records: &[Vec<MiniTransformerMlpWindowEvalRecord>],
    start: usize,
    end: usize,
) -> usize {
    (0..records.len())
        .min_by_key(|&expert| {
            let rows = &records[expert][start..end];
            (
                rows.iter()
                    .map(|row| row.probability_error_q15)
                    .sum::<usize>(),
                rows.iter().map(|row| row.mistakes).sum::<usize>(),
                expert,
            )
        })
        .unwrap_or(0)
}

fn aggregate_json(value: &Aggregate) -> String {
    let accuracy = value.windows.saturating_sub(value.mistakes) * 1000 / value.windows.max(1);
    let mean_error = value.probability_error_q15 / value.windows.max(1);
    let utilization = value
        .utilization_tokens
        .map(|count| count.saturating_mul(1000) / value.windows.max(1));
    format!(
        "{{\"windows\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{},\"invalid_forward_count\":{},\"decisions\":{},\"route_switches\":{},\"utilization_tokens\":[{},{},{}],\"utilization_per_mille\":[{},{},{}]}}",
        value.windows,
        value.mistakes,
        accuracy,
        value.probability_error_q15,
        mean_error,
        value.invalid_forward_count,
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

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
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

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

#[cfg(test)]
mod tests {
    use super::*;

    fn record(error: usize, mistake: usize) -> MiniTransformerMlpWindowEvalRecord {
        MiniTransformerMlpWindowEvalRecord {
            start: 0,
            end: 1,
            mistakes: mistake,
            probability_error_q15: error,
            invalid_forward_count: 0,
            predicted_token: Some(0),
            last_hidden_q15: [0; nsrl_train::MINI_TRANSFORMER_D_MODEL],
            router_hidden_features_q15: [0; 32],
            logits_q8: None,
        }
    }

    #[test]
    fn granularity_selection_uses_loss_then_mistakes_then_index() {
        let records = vec![
            vec![record(3, 1), record(8, 1)],
            vec![record(4, 0), record(2, 0)],
            vec![record(3, 0), record(7, 0)],
        ];
        assert_eq!(best_expert(&records, 0, 2), 1);
        assert_eq!(best_expert(&records, 0, 1), 2);
        assert_eq!(best_expert(&records, 1, 2), 1);
    }
}
