#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_eval::open_generation::{
    load_open_generation_development_panel, load_open_generation_manifest,
};
use nsrl_train::production::{
    ProductionModelV1, forward_production_model, forward_production_model_branch_hashes,
};

const SCHEMA: &str = "nsrl.production_residual_saturation_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-residual-saturation-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    manifest: PathBuf,
    tokenizer: PathBuf,
    model: PathBuf,
    trace: PathBuf,
}

#[derive(Debug)]
struct LayerHealth {
    attention: usize,
    mlp: usize,
}

#[derive(Debug)]
struct PromptHealth {
    id: String,
    tokens: usize,
    total: usize,
    layers: Vec<LayerHealth>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-residual-saturation-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let audit_binary_fnv64 = fnv64(&fs::read(env::current_exe()?)?);
    let manifest_bytes = fs::read(&config.manifest)?;
    let manifest = load_open_generation_manifest(&config.manifest)?;
    let prompts = load_open_generation_development_panel(&manifest)?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let model_bytes = fs::read(&config.model)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("audit model and tokenizer bindings do not match".into());
    }

    let mut audits = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let tokens = tokenizer.encode(&prompt.prompt);
        if tokens.is_empty() || tokens.len() > model.config.context_tokens {
            return Err(format!("prompt {} is outside the model context", prompt.id).into());
        }
        let forward = forward_production_model(&model, &tokens)?;
        let branches = forward_production_model_branch_hashes(&model, &tokens)?;
        if branches.layers.len() != model.config.layers {
            return Err("branch audit layer count does not match the model".into());
        }
        let layers = branches
            .layers
            .iter()
            .map(|layer| LayerHealth {
                attention: layer.attention_residual_saturation_count,
                mlp: layer.mlp_residual_saturation_count,
            })
            .collect::<Vec<_>>();
        let total = layers
            .iter()
            .map(|layer| layer.attention.saturating_add(layer.mlp))
            .sum::<usize>();
        if total != forward.residual_saturation_count {
            return Err("per-layer saturation counts do not reproduce the forward total".into());
        }
        audits.push(PromptHealth {
            id: prompt.id,
            tokens: tokens.len(),
            total,
            layers,
        });
    }

    let mut layer_totals = (0..model.config.layers)
        .map(|_| LayerHealth {
            attention: 0,
            mlp: 0,
        })
        .collect::<Vec<_>>();
    for audit in &audits {
        for (total, layer) in layer_totals.iter_mut().zip(&audit.layers) {
            total.attention = total.attention.saturating_add(layer.attention);
            total.mlp = total.mlp.saturating_add(layer.mlp);
        }
    }
    let attention_total = layer_totals
        .iter()
        .map(|layer| layer.attention)
        .sum::<usize>();
    let mlp_total = layer_totals.iter().map(|layer| layer.mlp).sum::<usize>();
    let total = attention_total.saturating_add(mlp_total);
    let saturated_prompts = audits.iter().filter(|audit| audit.total > 0).count();

    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"model_hash\":\"0x{:016x}\",\"tokenizer_hash\":\"0x{:016x}\",",
            "\"model_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"manifest_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"prompts\":{},\"layers\":{},\"saturated_prompts\":{}}},",
            "\"aggregate\":{{\"residual_saturation_count\":{},",
            "\"attention_residual_saturation_count\":{},",
            "\"mlp_residual_saturation_count\":{},\"layers\":["
        ),
        SCHEMA,
        model.model_hash(),
        model.tokenizer_hash,
        fnv64(&model_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&manifest_bytes),
        fnv64(AUDIT_SOURCE),
        audit_binary_fnv64,
        audits.len(),
        model.config.layers,
        saturated_prompts,
        total,
        attention_total,
        mlp_total,
    );
    push_layers(&mut output, &layer_totals)?;
    output.push_str("]},\"prompts\":[");
    for (index, audit) in audits.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":\"{}\",\"prompt_tokens\":{},\"residual_saturation_count\":{},\"layers\":[",
            audit.id, audit.tokens, audit.total,
        )?;
        push_layers(&mut output, &audit.layers)?;
        output.push_str("]}");
    }
    output.push_str("]}\n");
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn push_layers(output: &mut String, layers: &[LayerHealth]) -> Result<(), std::fmt::Error> {
    for (index, layer) in layers.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"layer\":{},\"attention\":{},\"mlp\":{}}}",
            index, layer.attention, layer.mlp,
        )?;
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut manifest = None;
    let mut tokenizer = None;
    let mut model = None;
    let mut trace = None;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Peekable<_>, option: &str| {
            args.next()
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match arg.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--model" => model = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--trace" => trace = Some(PathBuf::from(value(&mut args, &arg)?)),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    Ok(Config {
        manifest: manifest.ok_or("--manifest is required")?,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        model: model.ok_or("--model is required")?,
        trace: trace.ok_or("--trace is required")?,
    })
}
