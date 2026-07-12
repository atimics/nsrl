#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerPositionPolicy, evaluate_mini_transformer_mlp_model,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut out_path = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut attention_kind = MiniTransformerAttentionKind::Linear;
    let mut position_policy = MiniTransformerPositionPolicy::Nope;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => out_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--attention" => {
                attention_kind = match required(&mut args, "--attention")?.as_str() {
                    "base2-softmax" => MiniTransformerAttentionKind::Base2Softmax,
                    "linear" => MiniTransformerAttentionKind::Linear,
                    _ => return Err("--attention requires base2-softmax or linear".into()),
                }
            }
            "--position" => {
                position_policy = match required(&mut args, "--position")?.as_str() {
                    "learned-absolute" => MiniTransformerPositionPolicy::LearnedAbsolute,
                    "nope" => MiniTransformerPositionPolicy::Nope,
                    _ => return Err("--position requires learned-absolute or nope".into()),
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mini-transformer-eval --tokens PATH --model PATH [--out PATH] \
                     [--stride N] [--max-windows N] [--attention base2-softmax|linear] \
                     [--position learned-absolute|nope]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let trace = evaluate_mini_transformer_mlp_model(
        &tokens,
        &model,
        MiniTransformerMlpEvalConfig {
            seq_len: model.context_seq_len,
            stride,
            max_windows,
            attention_kind,
            position_policy,
        },
    )?
    .to_json_line();
    if let Some(path) = out_path {
        fs::write(path, trace)?;
    } else {
        print!("{trace}");
    }
    Ok(())
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}
