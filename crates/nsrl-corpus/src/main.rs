use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;

use nsrl_corpus::{CorpusConfig, prepare_corpus};

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
    let mut output_path = None;
    let mut trace_path = None;
    let mut max_simplewiki_pages = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "prepare" if command.is_none() => command = Some(arg),
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
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    if command.as_deref() != Some("prepare") {
        return Err("expected command: prepare".into());
    }

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

fn print_help() {
    println!(
        "Usage: nsrl-corpus prepare --shakespeare PATH --simplewiki-xml PATH|- --out PATH --trace PATH [--max-simplewiki-pages N]"
    );
    println!();
    println!("Prepares a deterministic Wiki-Bard text corpus and JSONL trace.");
}
