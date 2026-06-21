#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;

use nsrl_corpus::{
    CorpusConfig, LexemeInputProfile, LexemeTokenizeConfig, LexemeVocabProfile, TokenTextProfile,
    TokenizeConfig, clean_gutenberg_text, prepare_corpus, tokenize_corpus, tokenize_lexeme_corpus,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-corpus: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut command = None;
    let mut shakespeare_path = None;
    let mut simplewiki_xml_path = None;
    let mut corpus_path = None;
    let mut output_path = None;
    let mut tokens_output_path = None;
    let mut vocab_output_path = None;
    let mut trace_path = None;
    let mut max_simplewiki_pages = None;
    let mut tokenize_config = TokenizeConfig::default();
    let mut lexeme_config = LexemeTokenizeConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "prepare" if command.is_none() => command = Some(arg),
            "tokenize" if command.is_none() => command = Some(arg),
            "lexeme-tokenize" if command.is_none() => command = Some(arg),
            "clean-gutenberg" if command.is_none() => command = Some(arg),
            "--shakespeare" => {
                shakespeare_path = Some(PathBuf::from(
                    args.next().ok_or("--shakespeare requires a path")?,
                ));
            }
            "--simplewiki-xml" => {
                simplewiki_xml_path =
                    Some(args.next().ok_or("--simplewiki-xml requires a path or -")?);
            }
            "--out" => {
                output_path = Some(PathBuf::from(args.next().ok_or("--out requires a path")?));
            }
            "--corpus" => {
                corpus_path = Some(PathBuf::from(
                    args.next().ok_or("--corpus requires a path")?,
                ));
            }
            "--tokens-out" => {
                tokens_output_path = Some(PathBuf::from(
                    args.next().ok_or("--tokens-out requires a path")?,
                ));
            }
            "--vocab-out" => {
                vocab_output_path = Some(PathBuf::from(
                    args.next().ok_or("--vocab-out requires a path")?,
                ));
            }
            "--trace" => {
                trace_path = Some(PathBuf::from(args.next().ok_or("--trace requires a path")?));
            }
            "--max-simplewiki-pages" => {
                max_simplewiki_pages = Some(
                    args.next()
                        .ok_or("--max-simplewiki-pages requires an integer")?
                        .parse()?,
                );
            }
            "--seq-len" => {
                let value = args
                    .next()
                    .ok_or("--seq-len requires an integer")?
                    .parse()?;
                tokenize_config.seq_len = value;
                lexeme_config.seq_len = value;
            }
            "--stride" => {
                let value = args.next().ok_or("--stride requires an integer")?.parse()?;
                tokenize_config.stride = value;
                lexeme_config.stride = value;
            }
            "--max-windows" => {
                let value = args
                    .next()
                    .ok_or("--max-windows requires an integer")?
                    .parse()?;
                tokenize_config.max_windows = Some(value);
                lexeme_config.max_windows = Some(value);
            }
            "--preview-tokens" => {
                let value = args
                    .next()
                    .ok_or("--preview-tokens requires an integer")?
                    .parse()?;
                tokenize_config.preview_tokens = value;
                lexeme_config.preview_tokens = value;
            }
            "--max-vocab" => {
                lexeme_config.max_vocab = args
                    .next()
                    .ok_or("--max-vocab requires an integer")?
                    .parse()?;
            }
            "--lexeme-input-profile" => {
                let value = args
                    .next()
                    .ok_or("--lexeme-input-profile requires plain or shakespeare-gutenberg")?;
                lexeme_config.input_profile = match value.as_str() {
                    "plain" => LexemeInputProfile::Plain,
                    "shakespeare" | "shakespeare-gutenberg" => {
                        LexemeInputProfile::ShakespeareGutenberg
                    }
                    _ => {
                        return Err(
                            "--lexeme-input-profile requires plain or shakespeare-gutenberg".into(),
                        );
                    }
                };
            }
            "--lexeme-vocab-profile" => {
                let value = args
                    .next()
                    .ok_or("--lexeme-vocab-profile requires frequency or balanced")?;
                lexeme_config.vocab_profile = match value.as_str() {
                    "frequency" => LexemeVocabProfile::Frequency,
                    "balanced" => LexemeVocabProfile::Balanced,
                    _ => return Err("--lexeme-vocab-profile requires frequency or balanced".into()),
                };
            }
            "--lexeme-frequency-cap" => {
                lexeme_config.vocab_frequency_cap = args
                    .next()
                    .ok_or("--lexeme-frequency-cap requires an integer")?
                    .parse()?;
            }
            "--text-profile" => {
                let value = args
                    .next()
                    .ok_or("--text-profile requires identity or ascii-lower")?;
                tokenize_config.text_profile = match value.as_str() {
                    "identity" => TokenTextProfile::Identity,
                    "ascii-lower" => TokenTextProfile::AsciiLower,
                    _ => return Err("--text-profile requires identity or ascii-lower".into()),
                };
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    match command.as_deref() {
        Some("prepare") => run_prepare(
            shakespeare_path,
            simplewiki_xml_path,
            output_path,
            trace_path,
            max_simplewiki_pages,
        ),
        Some("tokenize") => {
            run_tokenize(corpus_path, tokens_output_path, trace_path, tokenize_config)
        }
        Some("lexeme-tokenize") => run_lexeme_tokenize(
            corpus_path,
            tokens_output_path,
            vocab_output_path,
            trace_path,
            lexeme_config,
        ),
        Some("clean-gutenberg") => run_clean_gutenberg(corpus_path, output_path),
        _ => Err("expected command: prepare, tokenize, lexeme-tokenize, or clean-gutenberg".into()),
    }
}

fn run_prepare(
    shakespeare_path: Option<PathBuf>,
    simplewiki_xml_path: Option<String>,
    output_path: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    max_simplewiki_pages: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let shakespeare_path = shakespeare_path.ok_or("--shakespeare is required")?;
    let simplewiki_xml_path = simplewiki_xml_path.ok_or("--simplewiki-xml is required")?;
    let output_path = output_path.ok_or("--out is required")?;
    let trace_path = trace_path.ok_or("--trace is required")?;

    let mut shakespeare = File::open(shakespeare_path)?;
    let mut output = BufWriter::new(File::create(output_path)?);
    let trace = if simplewiki_xml_path == "-" {
        let stdin = io::stdin();
        let reader = stdin.lock();
        prepare_corpus(
            &mut shakespeare,
            reader,
            &mut output,
            CorpusConfig {
                max_simplewiki_pages,
            },
        )?
    } else {
        let simplewiki = File::open(simplewiki_xml_path)?;
        let reader = BufReader::new(simplewiki);
        prepare_corpus(
            &mut shakespeare,
            reader,
            &mut output,
            CorpusConfig {
                max_simplewiki_pages,
            },
        )?
    };

    std::fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn run_tokenize(
    corpus_path: Option<PathBuf>,
    tokens_output_path: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    config: TokenizeConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = corpus_path.ok_or("--corpus is required")?;
    let tokens_output_path = tokens_output_path.ok_or("--tokens-out is required")?;
    let trace_path = trace_path.ok_or("--trace is required")?;

    let mut corpus = File::open(corpus_path)?;
    let mut tokens_out = BufWriter::new(File::create(tokens_output_path)?);
    let trace = tokenize_corpus(&mut corpus, &mut tokens_out, config)?;
    std::fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn run_lexeme_tokenize(
    corpus_path: Option<PathBuf>,
    tokens_output_path: Option<PathBuf>,
    vocab_output_path: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    config: LexemeTokenizeConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = corpus_path.ok_or("--corpus is required")?;
    let tokens_output_path = tokens_output_path.ok_or("--tokens-out is required")?;
    let vocab_output_path = vocab_output_path.ok_or("--vocab-out is required")?;
    let trace_path = trace_path.ok_or("--trace is required")?;

    let mut corpus = File::open(corpus_path)?;
    let mut tokens_out = BufWriter::new(File::create(tokens_output_path)?);
    let mut vocab_out = BufWriter::new(File::create(vocab_output_path)?);
    let trace = tokenize_lexeme_corpus(&mut corpus, &mut tokens_out, &mut vocab_out, config)?;
    std::fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn run_clean_gutenberg(
    corpus_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = corpus_path.ok_or("--corpus is required")?;
    let output_path = output_path.ok_or("--out is required")?;
    let input = fs::read_to_string(corpus_path)?;
    fs::write(output_path, clean_gutenberg_text(&input))?;
    Ok(())
}

fn print_help() {
    println!("Usage:");
    println!(
        "  nsrl-corpus prepare --shakespeare PATH --simplewiki-xml PATH|- --out PATH --trace PATH [--max-simplewiki-pages N]"
    );
    println!(
        "  nsrl-corpus tokenize --corpus PATH --tokens-out PATH --trace PATH [--seq-len N] [--stride N] [--max-windows N] [--preview-tokens N] [--text-profile identity|ascii-lower]"
    );
    println!(
        "  nsrl-corpus lexeme-tokenize --corpus PATH --tokens-out PATH --vocab-out PATH --trace PATH [--seq-len N] [--stride N] [--max-windows N] [--preview-tokens N] [--max-vocab N] [--lexeme-input-profile plain|shakespeare-gutenberg] [--lexeme-vocab-profile frequency|balanced] [--lexeme-frequency-cap N]"
    );
    println!("  nsrl-corpus clean-gutenberg --corpus PATH --out PATH");
    println!();
    println!(
        "Prepares deterministic Wiki-Bard corpus, byte-token traces, and lexeme-token traces."
    );
}
