#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::{SubwordTokenizer, SubwordTrainConfig};

#[derive(Debug, Default)]
struct Config {
    command: String,
    corpus: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    tokenizer_out: Option<PathBuf>,
    tokens: Option<PathBuf>,
    tokens_out: Option<PathBuf>,
    output: Option<PathBuf>,
    trace: Option<PathBuf>,
    vocab_size: Option<usize>,
    min_pair_frequency: Option<usize>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-subword: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    match config.command.as_str() {
        "train" => train(config),
        "encode" => encode(config),
        "decode" => decode(config),
        "inspect" => inspect(config),
        "help" => {
            print_help();
            Ok(())
        }
        _ => Err("expected train, encode, decode, or inspect".into()),
    }
}

fn train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = required_path(config.corpus, "--corpus")?;
    let tokenizer_path = required_path(config.tokenizer_out, "--tokenizer-out")?;
    let trace_path = required_path(config.trace, "--trace")?;
    let corpus = fs::read(corpus_path)?;
    let (tokenizer, trace) = SubwordTokenizer::train(
        &corpus,
        SubwordTrainConfig {
            target_vocab_size: config.vocab_size.unwrap_or(8_192),
            min_pair_frequency: config.min_pair_frequency.unwrap_or(2),
        },
    )?;
    fs::write(tokenizer_path, tokenizer.to_bytes())?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn encode(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = required_path(config.corpus, "--corpus")?;
    let tokenizer_path = required_path(config.tokenizer, "--tokenizer")?;
    let tokens_path = required_path(config.tokens_out, "--tokens-out")?;
    let trace_path = required_path(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let corpus = fs::read(corpus_path)?;
    let (tokens, trace) = tokenizer.encode_with_trace(&corpus);
    fs::write(tokens_path, tokenizer.token_stream_bytes(&tokens)?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn decode(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required_path(config.tokenizer, "--tokenizer")?;
    let tokens_path = required_path(config.tokens, "--tokens")?;
    let output_path = required_path(config.output, "--out")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let tokens = tokenizer.tokens_from_stream_bytes(&fs::read(tokens_path)?)?;
    fs::write(output_path, tokenizer.decode(&tokens)?)?;
    Ok(())
}

fn inspect(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required_path(config.tokenizer, "--tokenizer")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    println!(
        "{{\"schema\":\"nsrl.subword_tokenizer_inspect.v1\",\"tokenizer\":\"deterministic_byte_bpe_v1\",\"hash\":\"0x{:016x}\",\"source_hash\":\"0x{:016x}\",\"target_vocab_size\":{},\"vocab_size\":{},\"merges\":{},\"min_pair_frequency\":{}}}",
        tokenizer.tokenizer_hash(),
        tokenizer.source_hash(),
        tokenizer.target_vocab_size(),
        tokenizer.vocab_size(),
        tokenizer.merges().len(),
        tokenizer.min_pair_frequency(),
    );
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "train" | "encode" | "decode" | "inspect" if config.command.is_empty() => {
                config.command = arg
            }
            "--corpus" => config.corpus = Some(next_path(&mut args, "--corpus")?),
            "--tokenizer" => config.tokenizer = Some(next_path(&mut args, "--tokenizer")?),
            "--tokenizer-out" => {
                config.tokenizer_out = Some(next_path(&mut args, "--tokenizer-out")?)
            }
            "--tokens" => config.tokens = Some(next_path(&mut args, "--tokens")?),
            "--tokens-out" => config.tokens_out = Some(next_path(&mut args, "--tokens-out")?),
            "--out" => config.output = Some(next_path(&mut args, "--out")?),
            "--trace" => config.trace = Some(next_path(&mut args, "--trace")?),
            "--vocab-size" => config.vocab_size = Some(next_usize(&mut args, "--vocab-size")?),
            "--min-pair-frequency" => {
                config.min_pair_frequency = Some(next_usize(&mut args, "--min-pair-frequency")?)
            }
            "--help" | "-h" => config.command = "help".to_string(),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(
        args.next()
            .ok_or_else(|| format!("{option} requires a path"))?,
    ))
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(args
        .next()
        .ok_or_else(|| format!("{option} requires an integer"))?
        .parse()?)
}

fn required_path(
    value: Option<PathBuf>,
    option: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("{option} is required").into())
}

fn print_help() {
    println!(
        "Usage:\n  nsrl-subword train --corpus PATH --tokenizer-out PATH --trace PATH [--vocab-size N] [--min-pair-frequency N]\n  nsrl-subword encode --corpus PATH --tokenizer PATH --tokens-out PATH --trace PATH\n  nsrl-subword decode --tokens PATH --tokenizer PATH --out PATH\n  nsrl-subword inspect --tokenizer PATH"
    );
}
