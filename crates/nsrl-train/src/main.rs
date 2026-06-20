use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use nsrl_train::{SoftmaxTrainConfig, TrainConfig, run_softmax_training, run_training_smoke};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = TrainConfig::default();
    let mut softmax_config = SoftmaxTrainConfig::default();
    let mut mode = String::from("softmax");
    let mut trace_path = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                mode = args.next().ok_or("--mode requires perceptron or softmax")?;
            }
            "--epochs" => {
                let epochs = args
                    .next()
                    .ok_or("--epochs requires a following integer")?
                    .parse()?;
                config.epochs = epochs;
                softmax_config.epochs = epochs;
            }
            "--learning-rate" => {
                let value = args
                    .next()
                    .ok_or("--learning-rate requires a following integer")?
                    .parse()?;
                config.learning_rate = value;
                softmax_config.learning_rate = i32::from(value);
            }
            "--lr-shift" => {
                softmax_config.learning_rate_shift = args
                    .next()
                    .ok_or("--lr-shift requires a following integer")?
                    .parse()?;
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
        "Usage: nsrl-train [--mode softmax|perceptron] [--epochs N] [--learning-rate N] [--lr-shift N] [--trace PATH]"
    );
    println!();
    println!("Runs a deterministic integer output-head training trace.");
}
