#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_eval::contract::{check_proof_results, proof_contract_json_line};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-eval: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("contract") => {
            if args.next().is_some() {
                return Err("contract takes no arguments".into());
            }
            print!("{}", proof_contract_json_line());
        }
        Some("check") => {
            let mut results = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--results" => results = Some(PathBuf::from(required(&mut args, "--results")?)),
                    other => return Err(format!("unknown check argument: {other}").into()),
                }
            }
            let input = fs::read_to_string(results.ok_or("check requires --results PATH")?)?;
            let check = check_proof_results(&input)?;
            print!("{}", check.to_json_line());
            if !check.passed {
                std::process::exit(1);
            }
        }
        Some("--help" | "-h") | None => print_help(),
        Some(other) => return Err(format!("unknown command: {other}").into()),
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

fn print_help() {
    println!(
        "Usage:\n  nsrl-eval contract\n  nsrl-eval check --results PATH\n\nThe check command exits 0 only when the integer candidate beats retrieval, byte-ngram, and float-reference baselines under the frozen proof contract."
    );
}
