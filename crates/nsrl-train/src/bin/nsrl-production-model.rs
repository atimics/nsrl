#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_train::production::{
    ProductionModelConfig, ProductionModelV1, ProductionSmokeConfig, decode_bound_token_stream,
    train_production_output_smoke,
};

#[derive(Debug)]
struct Config {
    command: String,
    profile: String,
    tokenizer: Option<PathBuf>,
    tokens: Option<PathBuf>,
    model: Option<PathBuf>,
    model_out: Option<PathBuf>,
    trace: Option<PathBuf>,
    seed: u64,
    context_tokens: usize,
    max_windows: usize,
    epochs: usize,
    feature_shift: u8,
    bias_step_q8: i32,
    margin_q8: i32,
}

impl Default for Config {
    fn default() -> Self {
        let smoke = ProductionSmokeConfig::default();
        Self {
            command: String::new(),
            profile: "p10m".to_string(),
            tokenizer: None,
            tokens: None,
            model: None,
            model_out: None,
            trace: None,
            seed: 7,
            context_tokens: smoke.context_tokens,
            max_windows: smoke.max_windows,
            epochs: smoke.epochs,
            feature_shift: smoke.feature_shift,
            bias_step_q8: smoke.bias_step_q8,
            margin_q8: smoke.margin_q8,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-model: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    match config.command.as_str() {
        "init" => init(config),
        "inspect" => inspect(config),
        "smoke-train" => smoke_train(config),
        "help" => {
            print_help();
            Ok(())
        }
        _ => Err("expected init, inspect, or smoke-train".into()),
    }
}

fn init(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model_config = ProductionModelConfig::profile(&config.profile)
        .ok_or("--profile requires p10m, p20m, or p30m")?;
    if tokenizer.vocab_size() != model_config.vocab_size {
        return Err("tokenizer vocabulary does not match production profile".into());
    }
    let model =
        ProductionModelV1::new_initial(model_config, tokenizer.tokenizer_hash(), config.seed)?;
    let bytes = model.try_to_bytes()?;
    fs::write(model_out, &bytes)?;
    fs::write(
        trace_path,
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_init.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"tokenizer_hash\":\"0x{:016x}\",\"initialization_seed\":{},",
                "\"model_hash\":\"0x{:016x}\",\"artifact_bytes\":{},",
                "\"full_layer_backward_ready\":false,\"output_head_smoke_ready\":true}}\n"
            ),
            model.config.profile_id().unwrap_or("custom"),
            model.parameter_count(),
            model.tokenizer_hash,
            model.initialization_seed,
            model.model_hash(),
            bytes.len(),
        ),
    )?;
    Ok(())
}

fn inspect(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = required(config.model, "--model")?;
    let bytes = fs::read(model_path)?;
    let model = ProductionModelV1::from_bytes(&bytes)?;
    println!(
        concat!(
            "{{\"schema\":\"nsrl.production_model_inspect.v1\",",
            "\"profile\":\"{}\",\"vocab_size\":{},\"d_model\":{},\"heads\":{},",
            "\"layers\":{},\"hidden_dim\":{},\"context_tokens\":{},",
            "\"parameter_count\":{},\"tokenizer_hash\":\"0x{:016x}\",",
            "\"initialization_seed\":{},\"model_hash\":\"0x{:016x}\",\"artifact_bytes\":{}}}"
        ),
        model.config.profile_id().unwrap_or("custom"),
        model.config.vocab_size,
        model.config.d_model,
        model.config.heads,
        model.config.layers,
        model.config.hidden_dim,
        model.config.context_tokens,
        model.parameter_count(),
        model.tokenizer_hash,
        model.initialization_seed,
        model.model_hash(),
        bytes.len(),
    );
    Ok(())
}

fn smoke_train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = train_production_output_smoke(
        &mut model,
        &tokens,
        token_stream_hash,
        ProductionSmokeConfig {
            context_tokens: config.context_tokens,
            max_windows: config.max_windows,
            epochs: config.epochs,
            feature_shift: config.feature_shift,
            bias_step_q8: config.bias_step_q8,
            margin_q8: config.margin_q8,
        },
    )?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "init" | "inspect" | "smoke-train" if config.command.is_empty() => config.command = arg,
            "--profile" => config.profile = next(&mut args, "--profile")?,
            "--tokenizer" => {
                config.tokenizer = Some(PathBuf::from(next(&mut args, "--tokenizer")?))
            }
            "--tokens" => config.tokens = Some(PathBuf::from(next(&mut args, "--tokens")?)),
            "--model" => config.model = Some(PathBuf::from(next(&mut args, "--model")?)),
            "--model-out" => {
                config.model_out = Some(PathBuf::from(next(&mut args, "--model-out")?))
            }
            "--trace" => config.trace = Some(PathBuf::from(next(&mut args, "--trace")?)),
            "--seed" => config.seed = next(&mut args, "--seed")?.parse()?,
            "--context-tokens" => {
                config.context_tokens = next(&mut args, "--context-tokens")?.parse()?
            }
            "--max-windows" => config.max_windows = next(&mut args, "--max-windows")?.parse()?,
            "--epochs" => config.epochs = next(&mut args, "--epochs")?.parse()?,
            "--feature-shift" => {
                config.feature_shift = next(&mut args, "--feature-shift")?.parse()?
            }
            "--bias-step-q8" => config.bias_step_q8 = next(&mut args, "--bias-step-q8")?.parse()?,
            "--margin-q8" => config.margin_q8 = next(&mut args, "--margin-q8")?.parse()?,
            "--help" | "-h" => config.command = "help".to_string(),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

fn next(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn required(value: Option<PathBuf>, option: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("{option} is required").into())
}

fn print_help() {
    println!(
        "Usage:\n  nsrl-production-model init --profile p10m|p20m|p30m --tokenizer PATH --model-out PATH --trace PATH [--seed N]\n  nsrl-production-model inspect --model PATH\n  nsrl-production-model smoke-train --tokenizer PATH --tokens PATH --model PATH --model-out PATH --trace PATH [--context-tokens N] [--max-windows N] [--epochs N] [--feature-shift N] [--bias-step-q8 N] [--margin-q8 N]"
    );
}
