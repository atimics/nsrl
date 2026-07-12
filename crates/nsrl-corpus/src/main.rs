#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;

use nsrl_corpus::{
    CorpusConfig, TokenTextProfile, TokenizeConfig, clean_gutenberg_text,
    extract_simplewiki_corpus, prepare_corpus, tokenize_corpus,
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
    let mut trace_path = None;
    let mut max_simplewiki_pages = None;
    let mut tokenize_config = TokenizeConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "prepare" if command.is_none() => command = Some(arg),
            "tokenize" if command.is_none() => command = Some(arg),
            "clean-gutenberg" if command.is_none() => command = Some(arg),
            "extract-simplewiki" if command.is_none() => command = Some(arg),
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
            }
            "--stride" => {
                let value = args.next().ok_or("--stride requires an integer")?.parse()?;
                tokenize_config.stride = value;
            }
            "--max-windows" => {
                let value = args
                    .next()
                    .ok_or("--max-windows requires an integer")?
                    .parse()?;
                tokenize_config.max_windows = Some(value);
            }
            "--preview-tokens" => {
                let value = args
                    .next()
                    .ok_or("--preview-tokens requires an integer")?
                    .parse()?;
                tokenize_config.preview_tokens = value;
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
        Some("clean-gutenberg") => run_clean_gutenberg(corpus_path, output_path),
        Some("extract-simplewiki") => run_extract_simplewiki(
            simplewiki_xml_path,
            output_path,
            trace_path,
            max_simplewiki_pages,
        ),
        _ => Err(
            "expected command: prepare, tokenize, extract-simplewiki, or clean-gutenberg".into(),
        ),
    }
}

fn run_extract_simplewiki(
    simplewiki_xml_path: Option<String>,
    output_path: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    max_simplewiki_pages: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let simplewiki_xml_path = simplewiki_xml_path.ok_or("--simplewiki-xml is required")?;
    let output_path = output_path.ok_or("--out is required")?;
    let trace_path = trace_path.ok_or("--trace is required")?;
    let mut output = BufWriter::new(File::create(output_path)?);

    let trace = if simplewiki_xml_path == "-" {
        let stdin = io::stdin();
        let reader = stdin.lock();
        extract_simplewiki_corpus(
            reader,
            &mut output,
            CorpusConfig {
                max_simplewiki_pages,
            },
        )?
    } else {
        let simplewiki = File::open(simplewiki_xml_path)?;
        let reader = BufReader::new(simplewiki);
        extract_simplewiki_corpus(
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
        "  nsrl-corpus extract-simplewiki --simplewiki-xml PATH|- --out PATH --trace PATH [--max-simplewiki-pages N]"
    );
    println!(
        "  nsrl-corpus tokenize --corpus PATH --tokens-out PATH --trace PATH [--seq-len N] [--stride N] [--max-windows N] [--preview-tokens N] [--text-profile identity|ascii-lower]"
    );
    println!("  nsrl-corpus clean-gutenberg --corpus PATH --out PATH");
    println!();
    println!("Prepares deterministic Wiki-Bard corpus and byte-token traces.");
}
