#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::mt6::{MiniTransformerV6Model, Mt6OverfitConfig, train_mt6_output_overfit};

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("nsrl-mt6-overfit: {error}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_out_path = None;
    let mut trace_out_path = None;
    let mut seq_len = 64_usize;
    let mut windows = 256_usize;
    let mut epochs = 64_usize;
    let mut min_accuracy_per_mille = 900_usize;
    let mut max_residual_saturations_per_window = 4096_usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--model-out" => model_out_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--trace-out" => trace_out_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--seq-len" => seq_len = required(&mut args, &arg)?.parse()?,
            "--windows" => windows = required(&mut args, &arg)?.parse()?,
            "--epochs" => epochs = required(&mut args, &arg)?.parse()?,
            "--min-accuracy-per-mille" => {
                min_accuracy_per_mille = required(&mut args, &arg)?.parse()?
            }
            "--max-residual-saturations-per-window" => {
                max_residual_saturations_per_window = required(&mut args, &arg)?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mt6-overfit --tokens PATH --model-out PATH [--trace-out PATH] \
                     [--seq-len N] [--windows N] [--epochs N] \
                     [--min-accuracy-per-mille N] \
                     [--max-residual-saturations-per-window N]"
                );
                return Ok(true);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if seq_len < 4
        || windows == 0
        || epochs == 0
        || min_accuracy_per_mille > 1000
        || max_residual_saturations_per_window == 0
    {
        return Err("invalid MT6 overfit configuration".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let trainable = tokens.len().saturating_sub(seq_len);
    if trainable == 0 {
        return Err("token file is too short for the requested sequence length".into());
    }
    let windows = windows.min(trainable);
    let mut contexts = Vec::with_capacity(windows);
    let mut targets = Vec::with_capacity(windows);
    for index in 0..windows {
        let start = if windows == 1 {
            0
        } else {
            index * (trainable - 1) / (windows - 1)
        };
        contexts.push(tokens[start..start + seq_len].to_vec());
        targets.push(tokens[start + seq_len]);
    }

    let mut model = MiniTransformerV6Model::new_initial(seq_len)?;
    let trace = train_mt6_output_overfit(
        &mut model,
        &contexts,
        &targets,
        Mt6OverfitConfig {
            epochs,
            ..Mt6OverfitConfig::default()
        },
    )?;
    let artifact = model.try_to_bytes()?;
    let decoded = MiniTransformerV6Model::from_bytes(&artifact)?;
    if decoded != model {
        return Err("NSRLMT6 artifact replay mismatch".into());
    }
    let model_out_path = model_out_path.ok_or("--model-out is required")?;
    fs::write(&model_out_path, &artifact)?;
    let residual_per_window = trace.residual_saturation_count / trace.windows.max(1);
    let passed = trace.final_accuracy_per_mille >= min_accuracy_per_mille
        && residual_per_window <= max_residual_saturations_per_window
        && trace.weight_saturation_count == 0;
    let base = trace.to_json_line();
    let line = format!(
        "{}{}\n",
        base.trim_end(),
        format_args!(
            "\n{{\"schema\":\"nsrl.mt6.local_overfit_gate.v1\",\"passed\":{passed},\"policy\":{{\"min_accuracy_per_mille\":{min_accuracy_per_mille},\"max_residual_saturations_per_window\":{max_residual_saturations_per_window},\"require_zero_weight_saturation\":true}},\"metrics\":{{\"accuracy_per_mille\":{},\"residual_saturations_per_window\":{},\"weight_saturation_count\":{},\"artifact_bytes\":{},\"model_hash\":\"0x{:016x}\"}},\"artifact\":{}}}",
            trace.final_accuracy_per_mille,
            residual_per_window,
            trace.weight_saturation_count,
            artifact.len(),
            model.model_hash(),
            json_string(&model_out_path.to_string_lossy()),
        )
    );
    if let Some(path) = trace_out_path {
        fs::write(path, &line)?;
    } else {
        print!("{line}");
    }
    Ok(passed)
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch.is_control() => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}
