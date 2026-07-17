#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::{BOS_TOKEN_ID, PAD_TOKEN_ID, SubwordTokenizer};
use nsrl_eval::open_generation::{
    load_open_generation_development_panel, load_open_generation_manifest,
};
use nsrl_train::production::{ProductionDecoder, ProductionModelV1};

const SCHEMA: &str = "nsrl.production_context_sensitivity_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-context-sensitivity-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    manifest: PathBuf,
    tokenizer: PathBuf,
    model: PathBuf,
    trace: PathBuf,
    top_k: usize,
}

#[derive(Debug)]
struct RankedToken {
    token: usize,
    logit_q8: i32,
    probability_q15: i16,
}

#[derive(Debug)]
struct PromptAudit {
    id: String,
    prompt_tokens: usize,
    feature_hash: u64,
    logits_hash: u64,
    ranked: Vec<RankedToken>,
    selected_bias_q8: i32,
    selected_feature_contribution_q8: i32,
    next_greedy_token: usize,
    greedy_self_loop: bool,
    residual_saturation_count: usize,
    logits: Vec<i32>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-context-sensitivity-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let audit_binary_fnv64 = fnv64(&fs::read(env::current_exe()?)?);
    let manifest = load_open_generation_manifest(&config.manifest)?;
    let prompts = load_open_generation_development_panel(&manifest)?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let model_bytes = fs::read(&config.model)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
        || config.top_k == 0
        || config.top_k > model.config.vocab_size.saturating_sub(2)
    {
        return Err("audit configuration or model binding is invalid".into());
    }

    let mut audits = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode(&prompt.prompt);
        if prompt_tokens.is_empty() || prompt_tokens.len() > model.config.context_tokens {
            return Err(format!("prompt {} is outside the model context", prompt.id).into());
        }
        let mut decoder = ProductionDecoder::new(&model)?;
        let mut forward = None;
        let mut residual_saturation_count = 0_usize;
        for token in &prompt_tokens {
            let row = decoder.step(*token)?;
            residual_saturation_count =
                residual_saturation_count.saturating_add(row.residual_saturation_count);
            forward = Some(row);
        }
        let forward = forward.ok_or("prompt did not produce a forward row")?;
        let ranked = rank_tokens(&forward.logits_q8, &forward.probabilities_q15, config.top_k);
        let selected = ranked.first().ok_or("prompt has no eligible token")?;
        let selected_bias_q8 = model.output_bias_q8[selected.token];
        let next = decoder.step(selected.token as u32)?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(next.residual_saturation_count);
        let next_greedy_token = rank_tokens(&next.logits_q8, &next.probabilities_q15, 1)
            .first()
            .ok_or("feedback row has no eligible token")?
            .token;
        audits.push(PromptAudit {
            id: prompt.id,
            prompt_tokens: prompt_tokens.len(),
            feature_hash: hash_i16(&forward.features_q15),
            logits_hash: hash_i32(&forward.logits_q8),
            selected_bias_q8,
            selected_feature_contribution_q8: selected.logit_q8.saturating_sub(selected_bias_q8),
            next_greedy_token,
            greedy_self_loop: next_greedy_token == selected.token,
            residual_saturation_count,
            logits: forward.logits_q8,
            ranked,
        });
    }

    let mut bias_rank = model
        .output_bias_q8
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, _)| !is_masked(*token))
        .collect::<Vec<_>>();
    bias_rank.sort_unstable_by(|(left_token, left), (right_token, right)| {
        right.cmp(left).then_with(|| left_token.cmp(right_token))
    });
    let bias_top = bias_rank
        .first()
        .ok_or("model has no eligible output bias")?;

    let unique_greedy_tokens = unique_count(audits.iter().map(|row| row.ranked[0].token as u64));
    let unique_feature_hashes = unique_count(audits.iter().map(|row| row.feature_hash));
    let unique_logits_hashes = unique_count(audits.iter().map(|row| row.logits_hash));
    let greedy_self_loops = audits.iter().filter(|row| row.greedy_self_loop).count();
    let greedy_token = audits.first().map(|row| row.ranked[0].token);
    let shared_greedy_token =
        greedy_token.filter(|token| audits.iter().all(|row| row.ranked[0].token == *token));
    let shared_logit_range = shared_greedy_token.map(|token| {
        let minimum = audits
            .iter()
            .map(|row| row.logits[token])
            .min()
            .unwrap_or(0);
        let maximum = audits
            .iter()
            .map(|row| row.logits[token])
            .max()
            .unwrap_or(0);
        (minimum, maximum)
    });
    let margins = audits
        .iter()
        .map(|row| {
            row.ranked[0]
                .logit_q8
                .saturating_sub(row.ranked[1].logit_q8)
        })
        .collect::<Vec<_>>();
    let mut pair_count = 0_usize;
    let mut minimum_differing_logits = usize::MAX;
    let mut minimum_logit_l1 = u64::MAX;
    let mut maximum_logit_l1 = 0_u64;
    for left in 0..audits.len() {
        for right in left + 1..audits.len() {
            pair_count += 1;
            let mut differing = 0_usize;
            let mut l1 = 0_u64;
            for (&a, &b) in audits[left].logits.iter().zip(&audits[right].logits) {
                differing += usize::from(a != b);
                l1 = l1.saturating_add((i64::from(a) - i64::from(b)).unsigned_abs());
            }
            minimum_differing_logits = minimum_differing_logits.min(differing);
            minimum_logit_l1 = minimum_logit_l1.min(l1);
            maximum_logit_l1 = maximum_logit_l1.max(l1);
        }
    }

    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"model_hash\":\"0x{:016x}\",\"tokenizer_hash\":\"0x{:016x}\",",
            "\"model_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"manifest_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"prompts\":{},\"top_k\":{},\"prompt_pairs\":{}}},",
            "\"aggregate\":{{\"unique_greedy_tokens\":{},",
            "\"unique_feature_hashes\":{},\"unique_logits_hashes\":{},",
            "\"greedy_self_loops\":{},\"minimum_greedy_margin_q8\":{},",
            "\"maximum_greedy_margin_q8\":{},\"minimum_pair_differing_logits\":{},",
            "\"minimum_pair_logit_l1_q8\":{},\"maximum_pair_logit_l1_q8\":{},",
            "\"residual_saturation_count\":{}}},",
            "\"output_bias\":{{\"top_token\":{},\"top_bias_q8\":{}}},"
        ),
        SCHEMA,
        model.model_hash(),
        model.tokenizer_hash,
        fnv64(&model_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&fs::read(&config.manifest)?),
        fnv64(AUDIT_SOURCE),
        audit_binary_fnv64,
        audits.len(),
        config.top_k,
        pair_count,
        unique_greedy_tokens,
        unique_feature_hashes,
        unique_logits_hashes,
        greedy_self_loops,
        margins.iter().copied().min().unwrap_or(0),
        margins.iter().copied().max().unwrap_or(0),
        minimum_differing_logits,
        minimum_logit_l1,
        maximum_logit_l1,
        audits
            .iter()
            .map(|row| row.residual_saturation_count)
            .sum::<usize>(),
        bias_top.0,
        bias_top.1,
    );
    if let (Some(token), Some((minimum, maximum))) = (shared_greedy_token, shared_logit_range) {
        write!(
            output,
            "\"shared_greedy\":{{\"token\":{token},\"minimum_logit_q8\":{minimum},\"maximum_logit_q8\":{maximum}}},"
        )?;
    } else {
        output.push_str("\"shared_greedy\":null,");
    }
    output.push_str("\"prompts\":[");
    for (index, row) in audits.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            concat!(
                "{{\"id\":\"{}\",\"prompt_tokens\":{},",
                "\"feature_hash\":\"0x{:016x}\",\"logits_hash\":\"0x{:016x}\",",
                "\"greedy_margin_q8\":{},\"selected_bias_q8\":{},",
                "\"selected_feature_contribution_q8\":{},\"next_greedy_token\":{},",
                "\"greedy_self_loop\":{},\"residual_saturation_count\":{},\"top_tokens\":["
            ),
            row.id,
            row.prompt_tokens,
            row.feature_hash,
            row.logits_hash,
            row.ranked[0]
                .logit_q8
                .saturating_sub(row.ranked[1].logit_q8),
            row.selected_bias_q8,
            row.selected_feature_contribution_q8,
            row.next_greedy_token,
            row.greedy_self_loop,
            row.residual_saturation_count,
        )?;
        for (rank, token) in row.ranked.iter().enumerate() {
            if rank > 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"rank\":{},\"token\":{},\"logit_q8\":{},\"probability_q15\":{}}}",
                rank + 1,
                token.token,
                token.logit_q8,
                token.probability_q15,
            )?;
        }
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

fn rank_tokens(logits: &[i32], probabilities: &[i16], top_k: usize) -> Vec<RankedToken> {
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, _)| !is_masked(*token))
        .map(|(token, logit_q8)| RankedToken {
            token,
            logit_q8,
            probability_q15: probabilities[token],
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .logit_q8
            .cmp(&left.logit_q8)
            .then_with(|| left.token.cmp(&right.token))
    });
    ranked.truncate(top_k.min(ranked.len()));
    ranked
}

fn is_masked(token: usize) -> bool {
    matches!(token as u32, BOS_TOKEN_ID | PAD_TOKEN_ID)
}

fn unique_count(values: impl Iterator<Item = u64>) -> usize {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values.len()
}

fn hash_i16(values: &[i16]) -> u64 {
    hash_bytes(values.iter().flat_map(|value| value.to_le_bytes()))
}

fn hash_i32(values: &[i32]) -> u64 {
    hash_bytes(values.iter().flat_map(|value| value.to_le_bytes()))
}

fn hash_bytes(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

fn fnv64(bytes: &[u8]) -> u64 {
    hash_bytes(bytes.iter().copied())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut manifest = None;
    let mut tokenizer = None;
    let mut model = None;
    let mut trace = None;
    let mut top_k = 8_usize;
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
            "--top-k" => top_k = value(&mut args, &arg)?.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    Ok(Config {
        manifest: manifest.ok_or("--manifest is required")?,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        model: model.ok_or("--model is required")?,
        trace: trace.ok_or("--trace is required")?,
        top_k,
    })
}
