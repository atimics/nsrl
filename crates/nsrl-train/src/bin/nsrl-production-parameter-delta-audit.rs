#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_train::production::ProductionModelV1;

const SCHEMA: &str = "nsrl.production_parameter_delta_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-parameter-delta-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    source: PathBuf,
    candidate: PathBuf,
    trace: PathBuf,
}

#[derive(Debug, Clone, Copy, Default)]
struct Delta {
    changed: usize,
    l1: u64,
    maximum_absolute: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-parameter-delta-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let audit_binary_fnv64 = fnv64(&fs::read(env::current_exe()?)?);
    let source_bytes = fs::read(&config.source)?;
    let candidate_bytes = fs::read(&config.candidate)?;
    let source = ProductionModelV1::from_bytes(&source_bytes)?;
    let candidate = ProductionModelV1::from_bytes(&candidate_bytes)?;
    if source.config != candidate.config || source.tokenizer_hash != candidate.tokenizer_hash {
        return Err("source and candidate model bindings do not match".into());
    }

    let c = source.config;
    let matrix = c.d_model * c.d_model;
    let up_matrix = c.d_model * c.hidden_dim;
    let down_matrix = c.hidden_dim * c.d_model;
    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"source_model_hash\":\"0x{:016x}\",",
            "\"candidate_model_hash\":\"0x{:016x}\",",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"source_artifact_fnv64\":\"0x{:016x}\",",
            "\"candidate_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"parameters\":{},\"layers\":{}}},\"groups\":{{"
        ),
        SCHEMA,
        source.model_hash(),
        candidate.model_hash(),
        source.tokenizer_hash,
        fnv64(&source_bytes),
        fnv64(&candidate_bytes),
        fnv64(AUDIT_SOURCE),
        audit_binary_fnv64,
        source.parameter_count(),
        c.layers,
    );

    push_group(
        &mut output,
        "embeddings",
        delta_i16(&source.embeddings, &candidate.embeddings),
    )?;
    output.push(',');
    push_layered_i16(
        &mut output,
        "attention_rms",
        &source.attention_rms_weights,
        &candidate.attention_rms_weights,
        c.d_model,
        c.layers,
    )?;
    output.push(',');
    push_layered_i16(
        &mut output,
        "mlp_rms",
        &source.mlp_rms_weights,
        &candidate.mlp_rms_weights,
        c.d_model,
        c.layers,
    )?;
    output.push(',');
    push_group(
        &mut output,
        "final_rms",
        delta_i16(&source.final_rms_weights, &candidate.final_rms_weights),
    )?;
    for (name, left, right, width) in [
        ("q", &source.q_weights, &candidate.q_weights, matrix),
        ("k", &source.k_weights, &candidate.k_weights, matrix),
        ("v", &source.v_weights, &candidate.v_weights, matrix),
        ("o", &source.o_weights, &candidate.o_weights, matrix),
        ("up", &source.up_weights, &candidate.up_weights, up_matrix),
        (
            "gate",
            &source.gate_weights,
            &candidate.gate_weights,
            up_matrix,
        ),
        (
            "down",
            &source.down_weights,
            &candidate.down_weights,
            down_matrix,
        ),
    ] {
        output.push(',');
        push_layered_i8(&mut output, name, left, right, width, c.layers)?;
    }
    output.push(',');
    push_group(
        &mut output,
        "output",
        delta_i16(&source.output_weights, &candidate.output_weights),
    )?;
    output.push(',');
    push_group(
        &mut output,
        "bias",
        delta_i32(&source.output_bias_q8, &candidate.output_bias_q8),
    )?;
    output.push_str("}}\n");

    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn push_layered_i8(
    output: &mut String,
    name: &str,
    left: &[i8],
    right: &[i8],
    width: usize,
    layers: usize,
) -> Result<(), std::fmt::Error> {
    write!(output, "\"{name}\":{{\"total\":")?;
    push_delta(output, delta_i8(left, right))?;
    output.push_str(",\"layers\":[");
    for layer in 0..layers {
        if layer > 0 {
            output.push(',');
        }
        let range = layer * width..(layer + 1) * width;
        write!(output, "{{\"layer\":{layer},\"delta\":")?;
        push_delta(output, delta_i8(&left[range.clone()], &right[range]))?;
        output.push('}');
    }
    output.push_str("]}");
    Ok(())
}

fn push_layered_i16(
    output: &mut String,
    name: &str,
    left: &[i16],
    right: &[i16],
    width: usize,
    layers: usize,
) -> Result<(), std::fmt::Error> {
    write!(output, "\"{name}\":{{\"total\":")?;
    push_delta(output, delta_i16(left, right))?;
    output.push_str(",\"layers\":[");
    for layer in 0..layers {
        if layer > 0 {
            output.push(',');
        }
        let range = layer * width..(layer + 1) * width;
        write!(output, "{{\"layer\":{layer},\"delta\":")?;
        push_delta(output, delta_i16(&left[range.clone()], &right[range]))?;
        output.push('}');
    }
    output.push_str("]}");
    Ok(())
}

fn push_group(output: &mut String, name: &str, delta: Delta) -> Result<(), std::fmt::Error> {
    write!(output, "\"{name}\":")?;
    push_delta(output, delta)
}

fn push_delta(output: &mut String, delta: Delta) -> Result<(), std::fmt::Error> {
    write!(
        output,
        "{{\"changed\":{},\"l1\":{},\"maximum_absolute\":{}}}",
        delta.changed, delta.l1, delta.maximum_absolute,
    )
}

fn delta_i8(left: &[i8], right: &[i8]) -> Delta {
    delta(
        left.iter().map(|&value| i64::from(value)),
        right.iter().map(|&value| i64::from(value)),
    )
}

fn delta_i16(left: &[i16], right: &[i16]) -> Delta {
    delta(
        left.iter().map(|&value| i64::from(value)),
        right.iter().map(|&value| i64::from(value)),
    )
}

fn delta_i32(left: &[i32], right: &[i32]) -> Delta {
    delta(
        left.iter().map(|&value| i64::from(value)),
        right.iter().map(|&value| i64::from(value)),
    )
}

fn delta(left: impl Iterator<Item = i64>, right: impl Iterator<Item = i64>) -> Delta {
    left.zip(right)
        .fold(Delta::default(), |mut result, (a, b)| {
            let absolute = (a - b).unsigned_abs();
            result.changed = result.changed.saturating_add(usize::from(absolute > 0));
            result.l1 = result.l1.saturating_add(absolute);
            result.maximum_absolute = result.maximum_absolute.max(absolute);
            result
        })
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut source = None;
    let mut candidate = None;
    let mut trace = None;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Peekable<_>, option: &str| {
            args.next()
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match arg.as_str() {
            "--source" => source = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--candidate" => candidate = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--trace" => trace = Some(PathBuf::from(value(&mut args, &arg)?)),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    Ok(Config {
        source: source.ok_or("--source is required")?,
        candidate: candidate.ok_or("--candidate is required")?,
        trace: trace.ok_or("--trace is required")?,
    })
}
