#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_eval::contract::{
    check_proof_baselines, check_proof_results, load_proof_manifest, proof_contract_json_line,
};
use nsrl_eval::open_generation::{
    load_open_generation_manifest, open_generation_contract_json_line,
};
use nsrl_eval::successor::{
    check_successor_results, load_successor_manifest, successor_contract_json_line,
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
        Some("successor-contract") => {
            if args.next().is_some() {
                return Err("successor-contract takes no arguments".into());
            }
            print!("{}", successor_contract_json_line());
        }
        Some("successor-check") => {
            let mut manifest = None;
            let mut results = None;
            let mut evidence = None;
            let mut allow_falsification = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => {
                        manifest = Some(PathBuf::from(required(&mut args, "--manifest")?))
                    }
                    "--results" => results = Some(PathBuf::from(required(&mut args, "--results")?)),
                    "--evidence" => {
                        evidence = Some(PathBuf::from(required(&mut args, "--evidence")?))
                    }
                    "--allow-falsification" => allow_falsification = true,
                    other => {
                        return Err(format!("unknown successor-check argument: {other}").into());
                    }
                }
            }
            let manifest = load_successor_manifest(
                &manifest.ok_or("successor-check requires --manifest PATH")?,
            )?;
            let input =
                fs::read_to_string(results.ok_or("successor-check requires --results PATH")?)?;
            let evidence = fs::read(evidence.ok_or("successor-check requires --evidence PATH")?)?;
            let check = check_successor_results(&input, &evidence, &manifest)?;
            print!("{}", check.to_json_line());
            if !check.passed && !allow_falsification {
                std::process::exit(1);
            }
        }
        Some("successor-manifest") => {
            let manifest = single_path_option(&mut args, "--manifest")?;
            print!("{}", load_successor_manifest(&manifest)?.to_json_line());
        }
        Some("open-generation-contract") => {
            if args.next().is_some() {
                return Err("open-generation-contract takes no arguments".into());
            }
            print!("{}", open_generation_contract_json_line());
        }
        Some("open-generation-manifest") => {
            let manifest = single_path_option(&mut args, "--manifest")?;
            print!(
                "{}",
                load_open_generation_manifest(&manifest)?.to_json_line()
            );
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
        "Usage:\n  nsrl-eval contract\n  nsrl-eval manifest --manifest PATH\n  nsrl-eval check-baselines --manifest PATH --results PATH\n  nsrl-eval check --manifest PATH --results PATH\n  nsrl-eval successor-contract\n  nsrl-eval successor-manifest --manifest PATH\n  nsrl-eval successor-check --manifest PATH --results PATH --evidence PATH [--allow-falsification]\n  nsrl-eval open-generation-contract\n  nsrl-eval open-generation-manifest --manifest PATH\n\nThe frozen v1 check remains probability-error compatible for replay. Successor v2 binds the exact dataset, tokenizer, candidate, evaluator, runner, evidence, matrix, and per-system replays; it requires canonical integer NLL and a genuine float-transformer baseline. A valid losing trial exits 1 unless --allow-falsification is supplied; its JSON still reports passed:false."
    );
}
