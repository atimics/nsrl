#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_eval::q22_compositional::{
    blind_q22_evaluation, check_q22_predictions, encode_q22_dataset, load_q22_manifest,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-q22-compositional-eval: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("manifest") => {
            let manifest = single_path_option(&mut args, "--manifest")?;
            print!("{}", load_q22_manifest(&manifest)?.to_json_line());
        }
        Some("encode") => {
            let mut manifest = None;
            let mut dataset = None;
            let mut output = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => manifest = Some(path_arg(&mut args, "--manifest")?),
                    "--dataset" => dataset = Some(path_arg(&mut args, "--dataset")?),
                    "--out" => output = Some(path_arg(&mut args, "--out")?),
                    other => return Err(format!("unknown encode argument: {other}").into()),
                }
            }
            let manifest = load_q22_manifest(&manifest.ok_or("encode requires --manifest PATH")?)?;
            let encoded =
                encode_q22_dataset(&dataset.ok_or("encode requires --dataset PATH")?, &manifest)?;
            fs::write(output.ok_or("encode requires --out PATH")?, encoded)?;
        }
        Some("blind") => {
            let mut manifest = None;
            let mut evaluation = None;
            let mut output = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => manifest = Some(path_arg(&mut args, "--manifest")?),
                    "--eval" => evaluation = Some(path_arg(&mut args, "--eval")?),
                    "--out" => output = Some(path_arg(&mut args, "--out")?),
                    other => return Err(format!("unknown blind argument: {other}").into()),
                }
            }
            let manifest = load_q22_manifest(&manifest.ok_or("blind requires --manifest PATH")?)?;
            let blinded =
                blind_q22_evaluation(&evaluation.ok_or("blind requires --eval PATH")?, &manifest)?;
            fs::write(output.ok_or("blind requires --out PATH")?, blinded)?;
        }
        Some("check") => {
            let mut manifest = None;
            let mut evaluation = None;
            let mut predictions = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => manifest = Some(path_arg(&mut args, "--manifest")?),
                    "--eval" => evaluation = Some(path_arg(&mut args, "--eval")?),
                    "--predictions" => predictions = Some(path_arg(&mut args, "--predictions")?),
                    other => return Err(format!("unknown check argument: {other}").into()),
                }
            }
            let manifest = load_q22_manifest(&manifest.ok_or("check requires --manifest PATH")?)?;
            let check = check_q22_predictions(
                &evaluation.ok_or("check requires --eval PATH")?,
                &predictions.ok_or("check requires --predictions PATH")?,
                &manifest,
            )?;
            print!("{}", check.to_json_line());
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
    let path = path_arg(args, expected)?;
    if args.next().is_some() {
        return Err(format!("{expected} accepts one path").into());
    }
    Ok(path)
}

fn path_arg(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(
        args.next()
            .ok_or_else(|| format!("{option} requires a value"))?,
    ))
}

fn print_help() {
    println!(
        "Usage:\n  nsrl-q22-compositional-eval manifest --manifest PATH\n  nsrl-q22-compositional-eval encode --manifest PATH --dataset PATH --out PATH\n  nsrl-q22-compositional-eval blind --manifest PATH --eval PATH --out PATH\n  nsrl-q22-compositional-eval check --manifest PATH --eval PATH --predictions PATH"
    );
}
