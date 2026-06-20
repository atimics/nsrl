use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use nsrl_train::{
    ByteGenerationConfig, ByteSoftmaxModel, ByteSoftmaxTrainConfig, LinearBackwardConfig,
    SoftmaxTrainConfig, TrainConfig, generate_byte_softmax, run_byte_softmax_training_with_model,
    run_linear_backward_smoke, run_softmax_training, run_training_smoke,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = TrainConfig::default();
    let mut softmax_config = SoftmaxTrainConfig::default();
    let mut linear_backward_config = LinearBackwardConfig::default();
    let mut byte_softmax_config = ByteSoftmaxTrainConfig::default();
    let mut byte_generation_config = ByteGenerationConfig { max_new_tokens: 32 };
    let mut mode = String::from("softmax");
    let mut tokens_path = None;
    let mut model_path = None;
    let mut model_out_path = None;
    let mut prompt = Vec::new();
    let mut trace_path = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                mode = args.next().ok_or(
                    "--mode requires softmax, perceptron, linear-backward, byte-softmax, or byte-generate",
                )?;
            }
            "--epochs" => {
                let epochs = args
                    .next()
                    .ok_or("--epochs requires a following integer")?
                    .parse()?;
                config.epochs = epochs;
                softmax_config.epochs = epochs;
                byte_softmax_config.epochs = epochs;
            }
            "--learning-rate" => {
                let value: i32 = args
                    .next()
                    .ok_or("--learning-rate requires a following integer")?
                    .parse()?;
                config.learning_rate = i8::try_from(value)?;
                softmax_config.learning_rate = value;
                linear_backward_config.learning_rate = value;
                byte_softmax_config.learning_rate = value;
            }
            "--lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--lr-shift requires a following integer")?
                    .parse()?;
                softmax_config.learning_rate_shift = value;
                linear_backward_config.learning_rate_shift = value;
                byte_softmax_config.learning_rate_shift = value;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(
                    args.next().ok_or("--tokens requires a following path")?,
                ));
            }
            "--model" => {
                model_path = Some(PathBuf::from(
                    args.next().ok_or("--model requires a following path")?,
                ));
            }
            "--model-out" => {
                model_out_path = Some(PathBuf::from(
                    args.next().ok_or("--model-out requires a following path")?,
                ));
            }
            "--prompt" => {
                prompt = args
                    .next()
                    .ok_or("--prompt requires a following string")?
                    .into_bytes();
            }
            "--max-new-tokens" => {
                byte_generation_config.max_new_tokens = args
                    .next()
                    .ok_or("--max-new-tokens requires an integer")?
                    .parse()?;
            }
            "--seq-len" => {
                byte_softmax_config.seq_len = args
                    .next()
                    .ok_or("--seq-len requires an integer")?
                    .parse()?;
            }
            "--stride" => {
                byte_softmax_config.stride =
                    args.next().ok_or("--stride requires an integer")?.parse()?;
            }
            "--max-windows" => {
                byte_softmax_config.max_windows = Some(
                    args.next()
                        .ok_or("--max-windows requires an integer")?
                        .parse()?,
                );
            }
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next().ok_or("--trace requires a following path")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let line = match mode.as_str() {
        "perceptron" | "smoke" => run_training_smoke(config)?.to_json_line(),
        "softmax" => run_softmax_training(softmax_config)?.to_json_line(),
        "linear-backward" | "linear_backward" => {
            run_linear_backward_smoke(linear_backward_config)?.to_json_line()
        }
        "byte-softmax" | "byte_softmax" => {
            let path = tokens_path.ok_or("--tokens is required for byte-softmax mode")?;
            let tokens = fs::read(path)?;
            let run = run_byte_softmax_training_with_model(&tokens, byte_softmax_config)?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "byte-generate" | "byte_generate" => {
            let path = model_path.ok_or("--model is required for byte-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for byte-generate mode".into());
            }
            let model_bytes = fs::read(path)?;
            let model = ByteSoftmaxModel::from_bytes(&model_bytes)?;
            generate_byte_softmax(&model, &prompt, byte_generation_config)?.to_json_line()
        }
        other => return Err(format!("unknown mode: {other}").into()),
    };
    if let Some(path) = trace_path {
        fs::write(path, line)?;
    } else {
        io::stdout().write_all(line.as_bytes())?;
    }

    Ok(())
}

fn print_help() {
    println!(
        "Usage: nsrl-train [--mode softmax|perceptron|linear-backward|byte-softmax|byte-generate] [--tokens PATH] [--model PATH] [--model-out PATH] [--prompt TEXT] [--max-new-tokens N] [--epochs N] [--learning-rate N] [--lr-shift N] [--seq-len N] [--stride N] [--max-windows N] [--trace PATH]"
    );
    println!();
    println!("Runs a deterministic integer training trace.");
}
