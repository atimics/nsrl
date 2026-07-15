#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, SubwordTokenizer, SubwordTrainConfig};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Default)]
struct Config {
    command: String,
    corpus: Option<PathBuf>,
    index: Option<PathBuf>,
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
        "encode-indexed" => encode_indexed(config),
        "decode" => decode(config),
        "inspect" => inspect(config),
        "help" => {
            print_help();
            Ok(())
        }
        _ => Err("expected train, encode, decode, or inspect".into()),
    }
}

fn encode_indexed(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = required_path(config.corpus, "--corpus")?;
    let index_path = required_path(config.index, "--index")?;
    let tokenizer_path = required_path(config.tokenizer, "--tokenizer")?;
    let tokens_path = required_path(config.tokens_out, "--tokens-out")?;
    let trace_path = required_path(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let index_text = fs::read_to_string(index_path)?;
    let mut corpus = File::open(corpus_path)?;
    let mut tokens = Vec::new();
    let mut input_bytes = 0_usize;
    let mut input_hash = FNV_OFFSET;
    let mut documents = 0_usize;

    for (line_index, line) in index_text.lines().enumerate() {
        if line_index == 0 {
            if line != "schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256" {
                return Err("invalid indexed corpus header".into());
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 7 || fields[0] != "nsrl.production_corpus_record.v1" {
            return Err(format!("invalid indexed corpus row {}", line_index + 1).into());
        }
        let offset = fields[3].parse::<u64>()?;
        let byte_count = fields[4].parse::<usize>()?;
        let mut bytes = vec![0_u8; byte_count];
        corpus.seek(SeekFrom::Start(offset))?;
        corpus.read_exact(&mut bytes)?;
        if format!("0x{:016x}", hash_bytes(&bytes)) != fields[5] {
            return Err(format!("indexed corpus document hash mismatch: {}", fields[2]).into());
        }
        for &byte in &bytes {
            input_hash ^= u64::from(byte);
            input_hash = input_hash.wrapping_mul(FNV_PRIME);
        }
        input_bytes = input_bytes
            .checked_add(bytes.len())
            .ok_or("indexed corpus byte count overflow")?;
        tokens.push(BOS_TOKEN_ID);
        tokens.extend(tokenizer.encode(&bytes));
        tokens.push(EOS_TOKEN_ID);
        documents += 1;
    }
    if documents == 0 {
        return Err("indexed corpus contains no documents".into());
    }
    fs::write(&tokens_path, tokenizer.token_stream_bytes(&tokens)?)?;
    fs::write(
        trace_path,
        format!(
            concat!(
                "{{\"schema\":\"nsrl.subword_indexed_encode_trace.v1\",",
                "\"tokenizer_hash\":\"0x{:016x}\",\"vocab_size\":{},",
                "\"documents\":{},\"input_bytes\":{},\"input_hash\":\"0x{:016x}\",",
                "\"tokens\":{},\"token_hash\":\"0x{:016x}\",",
                "\"bos_tokens\":{},\"eos_tokens\":{},",
                "\"tokens_per_input_byte_per_mille\":{}}}\n"
            ),
            tokenizer.tokenizer_hash(),
            tokenizer.vocab_size(),
            documents,
            input_bytes,
            input_hash,
            tokens.len(),
            hash_tokens(&tokens),
            documents,
            documents,
            tokens.len().saturating_mul(1000) / input_bytes.max(1),
        ),
    )?;
    Ok(())
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
            "train" | "encode" | "encode-indexed" | "decode" | "inspect"
                if config.command.is_empty() =>
            {
                config.command = arg
            }
            "--corpus" => config.corpus = Some(next_path(&mut args, "--corpus")?),
            "--index" => config.index = Some(next_path(&mut args, "--index")?),
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
        "Usage:\n  nsrl-subword train --corpus PATH --tokenizer-out PATH --trace PATH [--vocab-size N] [--min-pair-frequency N]\n  nsrl-subword encode --corpus PATH --tokenizer PATH --tokens-out PATH --trace PATH\n  nsrl-subword encode-indexed --corpus PATH --index PATH --tokenizer PATH --tokens-out PATH --trace PATH\n  nsrl-subword decode --tokens PATH --tokenizer PATH --out PATH\n  nsrl-subword inspect --tokenizer PATH"
    );
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |mut hash, &token| {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
}
