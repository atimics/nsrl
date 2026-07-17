#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;

use nsrl_train::production::{ProductionModelV1, ProductionOptimizerStateV2};

const SCHEMA: &str = "nsrl.production_optimizer_residual_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-optimizer-residual-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    model: PathBuf,
    optimizer: PathBuf,
    trace: PathBuf,
    group: String,
    shifts: Vec<u8>,
}

#[derive(Debug)]
struct GroupRange {
    range: Range<usize>,
    layer_width: Option<usize>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-optimizer-residual-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let audit_binary_fnv64 = fnv64(&fs::read(env::current_exe()?)?);
    let model_bytes = fs::read(&config.model)?;
    let optimizer_bytes = fs::read(&config.optimizer)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    let optimizer = ProductionOptimizerStateV2::from_bytes(&optimizer_bytes)?;
    if optimizer.bound_model_hash != model.model_hash()
        || optimizer.tokenizer_hash != model.tokenizer_hash
        || optimizer.residuals.len() != model.parameter_count()
    {
        return Err("optimizer state is not bound to the candidate model".into());
    }
    let group = group_range(&model, &config.group)?;
    let residuals = &optimizer.residuals[group.range.clone()];
    let mut absolute = residuals
        .iter()
        .map(|value| value.unsigned_abs())
        .collect::<Vec<_>>();
    absolute.sort_unstable();
    let maximum = absolute.last().copied().unwrap_or(0);
    let nonzero = absolute.iter().filter(|&&value| value > 0).count();

    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"model_hash\":\"0x{:016x}\",\"optimizer_state_hash\":\"0x{:016x}\",",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"model_artifact_fnv64\":\"0x{:016x}\",",
            "\"optimizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"cursor\":{{\"step\":{},\"next_epoch\":{},\"next_window\":{}}},",
            "\"group\":{{\"name\":\"{}\",\"range\":[{},{}],\"parameters\":{},",
            "\"nonzero_residuals\":{},\"maximum_absolute_residual\":{},",
            "\"p99_absolute_residual\":{},\"p999_absolute_residual\":{},",
            "\"p9999_absolute_residual\":{},\"thresholds\":["
        ),
        SCHEMA,
        model.model_hash(),
        optimizer.state_hash(),
        model.tokenizer_hash,
        fnv64(&model_bytes),
        fnv64(&optimizer_bytes),
        fnv64(AUDIT_SOURCE),
        audit_binary_fnv64,
        optimizer.step,
        optimizer.next_epoch,
        optimizer.next_window,
        config.group,
        group.range.start,
        group.range.end,
        residuals.len(),
        nonzero,
        maximum,
        percentile(&absolute, 9900),
        percentile(&absolute, 9990),
        percentile(&absolute, 9999),
    );
    for (index, &shift) in config.shifts.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let threshold = 1_u64 << (shift - 1);
        let crossings = absolute.iter().filter(|&&value| value >= threshold).count();
        write!(
            output,
            "{{\"effective_shift\":{shift},\"nonzero_update_threshold\":{threshold},\"coordinates_at_threshold\":{crossings}}}",
        )?;
    }
    output.push(']');
    if let Some(width) = group.layer_width {
        output.push_str(",\"layers\":[");
        for layer in 0..model.config.layers {
            if layer > 0 {
                output.push(',');
            }
            let layer_residuals = &residuals[layer * width..(layer + 1) * width];
            let layer_maximum = layer_residuals
                .iter()
                .map(|value| value.unsigned_abs())
                .max()
                .unwrap_or(0);
            write!(
                output,
                "{{\"layer\":{layer},\"maximum_absolute_residual\":{layer_maximum},\"thresholds\":[",
            )?;
            for (index, &shift) in config.shifts.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                let threshold = 1_u64 << (shift - 1);
                let crossings = layer_residuals
                    .iter()
                    .filter(|value| value.unsigned_abs() >= threshold)
                    .count();
                write!(
                    output,
                    "{{\"effective_shift\":{shift},\"coordinates_at_threshold\":{crossings}}}",
                )?;
            }
            output.push_str("]}");
        }
        output.push(']');
    }
    output.push_str("}}\n");
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn group_range(model: &ProductionModelV1, name: &str) -> Result<GroupRange, &'static str> {
    let mut cursor = 0_usize;
    let mut take = |len: usize, layer_width| {
        let range = cursor..cursor + len;
        cursor += len;
        GroupRange { range, layer_width }
    };
    let groups = [
        ("embeddings", take(model.embeddings.len(), None)),
        (
            "attention_rms",
            take(
                model.attention_rms_weights.len(),
                Some(model.config.d_model),
            ),
        ),
        (
            "mlp_rms",
            take(model.mlp_rms_weights.len(), Some(model.config.d_model)),
        ),
        ("final_rms", take(model.final_rms_weights.len(), None)),
        (
            "q",
            take(
                model.q_weights.len(),
                Some(model.config.d_model * model.config.d_model),
            ),
        ),
        (
            "k",
            take(
                model.k_weights.len(),
                Some(model.config.d_model * model.config.d_model),
            ),
        ),
        (
            "v",
            take(
                model.v_weights.len(),
                Some(model.config.d_model * model.config.d_model),
            ),
        ),
        (
            "o",
            take(
                model.o_weights.len(),
                Some(model.config.d_model * model.config.d_model),
            ),
        ),
        (
            "up",
            take(
                model.up_weights.len(),
                Some(model.config.d_model * model.config.hidden_dim),
            ),
        ),
        (
            "gate",
            take(
                model.gate_weights.len(),
                Some(model.config.d_model * model.config.hidden_dim),
            ),
        ),
        (
            "down",
            take(
                model.down_weights.len(),
                Some(model.config.hidden_dim * model.config.d_model),
            ),
        ),
        ("output", take(model.output_weights.len(), None)),
        ("bias", take(model.output_bias_q8.len(), None)),
    ];
    if cursor != model.parameter_count() {
        return Err("model parameter ranges do not cover the artifact");
    }
    groups
        .into_iter()
        .find_map(|(group, range)| (group == name).then_some(range))
        .ok_or("unknown parameter group")
}

fn percentile(sorted: &[u64], basis_points: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = sorted
        .len()
        .saturating_mul(basis_points)
        .div_ceil(10_000)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut optimizer = None;
    let mut trace = None;
    let mut group = None;
    let mut shifts = Vec::<u8>::new();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Peekable<_>, option: &str| {
            args.next()
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--optimizer" => optimizer = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--trace" => trace = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--group" => group = Some(value(&mut args, &arg)?),
            "--shift" => shifts.push(value(&mut args, &arg)?.parse()?),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if shifts.is_empty() || shifts.iter().any(|&shift| !(1..=62).contains(&shift)) {
        return Err("at least one --shift in 1..=62 is required".into());
    }
    shifts.sort_unstable_by(|left, right| right.cmp(left));
    shifts.dedup();
    Ok(Config {
        model: model.ok_or("--model is required")?,
        optimizer: optimizer.ok_or("--optimizer is required")?,
        trace: trace.ok_or("--trace is required")?,
        group: group.ok_or("--group is required")?,
        shifts,
    })
}
