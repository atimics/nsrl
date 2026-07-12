#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerPositionPolicy, evaluate_mini_transformer_mlp_model,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-oracle-score: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input_path = None;
    let mut output_path = None;
    let mut model_specs = Vec::new();
    let mut stride = 8_usize;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input_path = Some(PathBuf::from(required(&mut args, "--input")?)),
            "--out" => output_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--model" => model_specs.push(required(&mut args, "--model")?),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mini-transformer-oracle-score --input PATH --out PATH \
                     --model ID=PATH --model ID=PATH --model ID=PATH [--stride N]\n\
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
    if stride == 0 {
        return Err("--stride must be positive".into());
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
        let model = MiniTransformerMlpModel::from_bytes(&fs::read(path)?)?;
        model_ids.push(id.to_string());
        models.push(model);
    }
    let seq_len = models[0].context_seq_len;
    if seq_len == 0 || models.iter().any(|model| model.context_seq_len != seq_len) {
        return Err("all models must use the same positive context length".into());
    }

    let input = BufReader::new(fs::File::open(input_path)?);
    let mut output = BufWriter::new(fs::File::create(output_path)?);
    writeln!(
        output,
        "sample_id\toracle_target\tchild_mean_probability_error_q15\tchild_accuracy_per_mille\tchild_ids\tchild_model_hashes"
    )?;
    let mut rows = 0_usize;
    for (line_index, line) in input.lines().enumerate() {
        let line = line?;
        if line_index == 0 && line == "sample_id\tprompt_hex" {
            continue;
        }
        let (sample_id, prompt_hex) = line
            .split_once('\t')
            .ok_or("input row requires sample_id and prompt_hex")?;
        if sample_id.is_empty() || sample_id.contains(['\t', ',']) {
            return Err("sample_id is invalid".into());
        }
        let prompt = decode_hex(prompt_hex)?;
        if prompt.len() <= seq_len {
            return Err(format!("prompt is too short for model context: {sample_id}").into());
        }

        let mut losses = Vec::with_capacity(3);
        let mut accuracies = Vec::with_capacity(3);
        let mut hashes = Vec::with_capacity(3);
        for model in &models {
            let trace = evaluate_mini_transformer_mlp_model(
                &prompt,
                model,
                MiniTransformerMlpEvalConfig {
                    seq_len,
                    stride,
                    max_windows: None,
                    attention_kind: MiniTransformerAttentionKind::Linear,
                    position_policy: MiniTransformerPositionPolicy::Nope,
                },
            )?;
            losses.push(trace.mean_probability_error_q15);
            accuracies.push(trace.accuracy_per_mille);
            hashes.push(trace.model_hash);
        }
        let oracle_target = (0..losses.len())
            .min_by_key(|&index| (losses[index], usize::MAX - accuracies[index], index))
            .ok_or("oracle scoring produced no candidates")?;
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}",
            sample_id,
            oracle_target,
            join_usize(&losses),
            join_usize(&accuracies),
            model_ids.join(","),
            hashes
                .iter()
                .map(|hash| format!("0x{hash:016x}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        rows += 1;
    }
    if rows == 0 {
        return Err("input contains no scoring rows".into());
    }
    output.flush()?;
    Ok(())
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
    let mut output = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
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

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_prompt_hex() {
        assert_eq!(decode_hex("436f646578").expect("hex"), b"Codex");
        assert!(decode_hex("").is_err());
        assert!(decode_hex("0").is_err());
        assert!(decode_hex("zz").is_err());
    }
}
