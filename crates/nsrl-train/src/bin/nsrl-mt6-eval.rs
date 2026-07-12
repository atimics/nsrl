#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::BYTE_VOCAB;
use nsrl_train::mt6::{MiniTransformerV6Model, forward_mt6};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mt6-eval: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut out_path = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--out" => out_path = Some(PathBuf::from(required(&mut args, &arg)?)),
            "--stride" => stride = required(&mut args, &arg)?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, &arg)?.parse()?),
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mt6-eval --tokens PATH --model PATH [--out PATH] \
                     [--stride N] [--max-windows N]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if stride == 0 || max_windows == Some(0) {
        return Err("stride and max-windows must be positive".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerV6Model::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let trainable = tokens.len().saturating_sub(model.context_seq_len);
    if trainable == 0 {
        return Err("token file is too short for this model".into());
    }
    let mut windows = 0_usize;
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    let mut residual_saturation_count = 0_usize;
    let mut prediction_counts = [0_usize; BYTE_VOCAB];
    for start in (0..trainable).step_by(stride) {
        if max_windows.is_some_and(|limit| windows >= limit) {
            break;
        }
        let end = start + model.context_seq_len;
        let target = tokens[end];
        let output = forward_mt6(&model, &tokens[start..end])?;
        let predicted = output
            .logits_q8
            .iter()
            .enumerate()
            .max_by_key(|&(token, logit)| (*logit, std::cmp::Reverse(token)))
            .map(|(token, _)| token)
            .ok_or("empty MT6 output")?;
        windows = windows.saturating_add(1);
        mistakes = mistakes.saturating_add(usize::from(predicted != usize::from(target)));
        prediction_counts[predicted] = prediction_counts[predicted].saturating_add(1);
        residual_saturation_count =
            residual_saturation_count.saturating_add(output.residual_saturation_count);
        probability_error_q15 = probability_error_q15.saturating_add(
            output
                .probabilities_q15
                .iter()
                .enumerate()
                .map(|(token, &probability)| {
                    if token == usize::from(target) {
                        (i32::from(i16::MAX) - i32::from(probability)).max(0) as usize
                    } else {
                        i32::from(probability).max(0) as usize
                    }
                })
                .sum::<usize>(),
        );
    }
    let unique_predictions = prediction_counts.iter().filter(|&&count| count > 0).count();
    let (most_predicted_token, most_predicted_count) = prediction_counts
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|&(token, count)| (count, std::cmp::Reverse(token)))
        .unwrap_or((0, 0));
    let line = format!(
        concat!(
            "{{\"schema\":\"nsrl.mt6.eval.v1\",",
            "\"model\":{{\"magic\":\"NSRLMT6\",\"hash\":\"0x{:016x}\",\"seq_len\":{}}},",
            "\"evaluation\":{{\"windows\":{},\"stride\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{},\"unique_predicted_tokens\":{},\"most_predicted_token\":{},\"most_predicted_token_count\":{},\"most_predicted_token_share_per_mille\":{},\"residual_saturations_per_window\":{}}}}}\n"
        ),
        model.model_hash(),
        model.context_seq_len,
        windows,
        stride,
        mistakes,
        windows.saturating_sub(mistakes) * 1000 / windows.max(1),
        probability_error_q15,
        probability_error_q15 / windows.max(1),
        unique_predictions,
        most_predicted_token,
        most_predicted_count,
        most_predicted_count * 1000 / windows.max(1),
        residual_saturation_count / windows.max(1),
    );
    if let Some(path) = out_path {
        fs::write(path, line)?;
    } else {
        print!("{line}");
    }
    Ok(())
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}
