#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_eval::contract::{
    check_proof_baselines, check_proof_results, load_proof_manifest, proof_contract_json_line,
};

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
        Some("manifest") => {
            let manifest = single_path_option(&mut args, "--manifest")?;
            print!("{}", load_proof_manifest(&manifest)?.to_json_line());
        }
        Some("check-baselines") => {
            let mut manifest = None;
            let mut results = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => {
                        manifest = Some(PathBuf::from(required(&mut args, "--manifest")?))
                    }
                    "--results" => results = Some(PathBuf::from(required(&mut args, "--results")?)),
                    other => {
                        return Err(format!("unknown check-baselines argument: {other}").into());
                    }
                }
            }
            let manifest =
                load_proof_manifest(&manifest.ok_or("check-baselines requires --manifest PATH")?)?;
            let input =
                fs::read_to_string(results.ok_or("check-baselines requires --results PATH")?)?;
            print!(
                "{}",
                check_proof_baselines(&input, &manifest)?.to_json_line()
            );
        }
        Some("check") => {
            let mut manifest = None;
            let mut results = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => {
                        manifest = Some(PathBuf::from(required(&mut args, "--manifest")?))
                    }
                    "--results" => results = Some(PathBuf::from(required(&mut args, "--results")?)),
                    other => return Err(format!("unknown check argument: {other}").into()),
                }
            }
            let manifest = load_proof_manifest(&manifest.ok_or("check requires --manifest PATH")?)?;
            let input = fs::read_to_string(results.ok_or("check requires --results PATH")?)?;
            let check = check_proof_results(&input)?;
            if check.dataset_hash != manifest.dataset_hash || check.targets != manifest.targets {
                return Err("proof results do not match the frozen manifest".into());
            }
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

fn single_path_option(
    args: &mut impl Iterator<Item = String>,
    expected: &'static str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if args.next().as_deref() != Some(expected) {
        return Err(format!("expected {expected} PATH").into());
    }
    let path = PathBuf::from(required(args, expected)?);
    if args.next().is_some() {
        return Err(format!("{expected} accepts one path").into());
    }
    Ok(path)
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
        "Usage:\n  nsrl-eval contract\n  nsrl-eval manifest --manifest PATH\n  nsrl-eval check-baselines --manifest PATH --results PATH\n  nsrl-eval check --manifest PATH --results PATH\n\nThe check command exits 0 only when the integer candidate beats retrieval, byte-ngram, and float-reference baselines under the frozen proof contract."
    );
}
