#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use nsrl_core::LinearKernel;
use nsrl_demo::{
    BenchmarkPreset, run_benchmark_suite_with_linear_kernel, run_benchmark_with_linear_kernel,
    run_forward,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-demo: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::from("abba");
    let mut trace_path = None;
    let mut preset = String::from("toy");
    let mut warmup_runs = 0_usize;
    let mut measured_runs = 1_usize;
    let mut linear_kernel = LinearKernel::GenericI8;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                input = args.next().ok_or("--input requires a following string")?;
            }
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next().ok_or("--trace requires a following path")?,
                ));
            }
            "--preset" => {
                preset = args.next().ok_or("--preset requires a following name")?;
            }
            "--warmup" => {
                warmup_runs = args
                    .next()
                    .ok_or("--warmup requires a following integer")?
                    .parse()?;
            }
            "--repeat" => {
                measured_runs = args
                    .next()
                    .ok_or("--repeat requires a following integer")?
                    .parse()?;
            }
            "--linear-kernel" => {
                linear_kernel = parse_linear_kernel(
                    &args
                        .next()
                        .ok_or("--linear-kernel requires generic or ternary")?,
                )?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let line = match preset.as_str() {
        "toy" => run_forward(&input)?.to_json_line(),
        "bench-1m" if warmup_runs == 0 && measured_runs == 1 => {
            run_benchmark_with_linear_kernel(BenchmarkPreset::bench_1m(), linear_kernel)?
                .to_json_line()
        }
        "bench-1m" => run_benchmark_suite_with_linear_kernel(
            BenchmarkPreset::bench_1m(),
            warmup_runs,
            measured_runs,
            linear_kernel,
        )?
        .to_json_line(),
        other => return Err(format!("unknown preset: {other}").into()),
    };
    if let Some(path) = trace_path {
        fs::write(path, line)?;
    } else {
        io::stdout().write_all(line.as_bytes())?;
    }

    Ok(())
}

fn parse_linear_kernel(value: &str) -> Result<LinearKernel, Box<dyn std::error::Error>> {
    match value {
        "generic" | "generic_i8" | "i8" => Ok(LinearKernel::GenericI8),
        "ternary" => Ok(LinearKernel::Ternary),
        #[cfg(target_arch = "aarch64")]
        "neon" | "neon_i8" => Ok(LinearKernel::NeonI8),
        other => Err(format!("unknown linear kernel: {other}").into()),
    }
}

fn print_help() {
    println!(
        "Usage: nsrl-demo [--preset toy|bench-1m] [--input TEXT] [--trace PATH] [--warmup N] [--repeat N] [--linear-kernel generic|ternary|neon_i8]"
    );
    println!();
    println!("Runs the fixed NSRL integer transformer demo or benchmark preset.");
    println!("Without --trace, writes one JSONL row to stdout.");
}
