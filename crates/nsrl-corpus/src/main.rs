use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;

use nsrl_corpus::{CorpusConfig, TokenizeConfig, prepare_corpus, tokenize_corpus};

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
                tokenize_config.seq_len = args
                    .next()
                    .ok_or("--seq-len requires an integer")?
                    .parse()?;
            }
            "--stride" => {
                tokenize_config.stride =
                    args.next().ok_or("--stride requires an integer")?.parse()?;
            }
            "--max-windows" => {
                tokenize_config.max_windows = Some(
                    args.next()
                        .ok_or("--max-windows requires an integer")?
                        .parse()?,
                );
            }
            "--preview-tokens" => {
                tokenize_config.preview_tokens = args
                    .next()
                    .ok_or("--preview-tokens requires an integer")?
                    .parse()?;
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
        _ => Err("expected command: prepare or tokenize".into()),
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

fn print_help() {
    println!("Usage:");
    println!(
        "  nsrl-corpus prepare --shakespeare PATH --simplewiki-xml PATH|- --out PATH --trace PATH [--max-simplewiki-pages N]"
    );
    println!(
        "  nsrl-corpus tokenize --corpus PATH --tokens-out PATH --trace PATH [--seq-len N] [--stride N] [--max-windows N] [--preview-tokens N]"
    );
    println!();
    println!("Prepares deterministic Wiki-Bard corpus and byte-token JSONL traces.");
}
