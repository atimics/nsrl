#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use nsrl_core::{
    DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, base2_exp_neg_q15, base2_softmax_nll_millibits,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, SubwordTokenizer};
use nsrl_eval::open_generation::{
    OPEN_GENERATION_CONTRACT_ID, OpenGenerationPrompt, load_open_generation_development_panel,
    load_open_generation_manifest,
};
use nsrl_train::production::{ProductionDecoder, ProductionModelV1};

const TRACE_SCHEMA: &str = "nsrl.open_generation_modeling.v1";
const RUNNER_SOURCE: &[u8] = include_bytes!("nsrl-open-generation-modeling-run.rs");
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
struct PromptScore {
    id: String,
    original_utf8_bytes: usize,
    candidate_tokens: usize,
    total_nll_millibits: u64,
    zero_probability_targets: usize,
    residual_saturation_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-open-generation-modeling-run: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let manifest_bytes = fs::read(&config.manifest)?;
    let manifest = load_open_generation_manifest(&config.manifest)?;
    let panel_bytes = fs::read(&manifest.development_panel_path)?;
    let prompts = load_open_generation_development_panel(&manifest)?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let model_bytes = fs::read(&config.model)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("candidate model and tokenizer binding mismatch".into());
    }

    let mut decoder = ProductionDecoder::new(&model)?;
    let cache_state_bytes = decoder.state_bytes();
    let cache_workspace_bytes = decoder.workspace_bytes();
    let mut scores = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        scores.push(score_prompt(&mut decoder, &tokenizer, prompt)?);
    }
    let original_utf8_bytes = scores
        .iter()
        .try_fold(0_u64, |total, score| {
            total.checked_add(score.original_utf8_bytes as u64)
        })
        .ok_or("original byte count overflow")?;
    let candidate_tokens = scores
        .iter()
        .try_fold(0_u64, |total, score| {
            total.checked_add(score.candidate_tokens as u64)
        })
        .ok_or("candidate token count overflow")?;
    let total_nll_millibits = scores
        .iter()
        .try_fold(0_u64, |total, score| {
            total.checked_add(score.total_nll_millibits)
        })
        .ok_or("modeling NLL overflow")?;
    let zero_probability_targets = scores
        .iter()
        .try_fold(0_usize, |total, score| {
            total.checked_add(score.zero_probability_targets)
        })
        .ok_or("zero-probability counter overflow")?;
    let residual_saturation_count = scores
        .iter()
        .try_fold(0_usize, |total, score| {
            total.checked_add(score.residual_saturation_count)
        })
        .ok_or("saturation counter overflow")?;
    if original_utf8_bytes == 0 || candidate_tokens == 0 {
        return Err("modeling panel must contain bytes and candidate tokens".into());
    }
    let millibits_per_original_utf8_byte = total_nll_millibits
        .checked_add(original_utf8_bytes / 2)
        .ok_or("modeling metric overflow")?
        / original_utf8_bytes;

    let mut trace = format!(
        concat!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",",
            "\"partition\":\"development\",",
            "\"objective\":\"integer_base2_softmax_nll_per_original_utf8_byte\",",
            "\"sequence_policy\":\"reset_per_prompt_bos_then_score_candidate_tokens_no_eos\",",
            "\"bindings\":{{\"manifest_fnv64\":\"0x{:016x}\",",
            "\"development_panel_fnv64\":\"0x{:016x}\",",
            "\"contract_tokenizer_fnv64\":\"{}\",",
            "\"candidate_tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"candidate_tokenizer_hash\":\"0x{:016x}\",",
            "\"candidate_model_artifact_fnv64\":\"0x{:016x}\",",
            "\"candidate_model_hash\":\"0x{:016x}\",",
            "\"runner_source_fnv64\":\"0x{:016x}\",",
            "\"runner_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"prompts\":{},\"original_utf8_bytes\":{},",
            "\"candidate_tokens\":{},\"zero_probability_targets\":{}}},",
            "\"metrics\":{{\"total_nll_millibits\":{},",
            "\"millibits_per_original_utf8_byte\":{},",
            "\"zero_probability_floor_millibits\":{}}},",
            "\"cache\":{{\"state_bytes\":{},\"workspace_bytes\":{}}},",
            "\"residual_saturation_count\":{},\"prompts\":["
        ),
        TRACE_SCHEMA,
        OPEN_GENERATION_CONTRACT_ID,
        fnv64(&manifest_bytes),
        fnv64(&panel_bytes),
        manifest.tokenizer_hash,
        fnv64(&tokenizer_bytes),
        tokenizer.tokenizer_hash(),
        fnv64(&model_bytes),
        model.model_hash(),
        fnv64(RUNNER_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        scores.len(),
        original_utf8_bytes,
        candidate_tokens,
        zero_probability_targets,
        total_nll_millibits,
        millibits_per_original_utf8_byte,
        DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
        cache_state_bytes,
        cache_workspace_bytes,
        residual_saturation_count,
    );
    for (index, score) in scores.iter().enumerate() {
        if index != 0 {
            trace.push(',');
        }
        let prompt_millibits_per_byte = score
            .total_nll_millibits
            .saturating_add(score.original_utf8_bytes as u64 / 2)
            / score.original_utf8_bytes as u64;
        write!(
            trace,
            concat!(
                "{{\"id\":\"{}\",\"original_utf8_bytes\":{},",
                "\"candidate_tokens\":{},\"total_nll_millibits\":{},",
                "\"millibits_per_original_utf8_byte\":{},",
                "\"zero_probability_targets\":{},",
                "\"residual_saturation_count\":{}}}"
            ),
            score.id,
            score.original_utf8_bytes,
            score.candidate_tokens,
            score.total_nll_millibits,
            prompt_millibits_per_byte,
            score.zero_probability_targets,
            score.residual_saturation_count,
        )?;
    }
    trace.push_str("]}\n");
    write_file(&config.trace, trace.as_bytes())?;
    print!("{trace}");
    Ok(())
}

fn score_prompt(
    decoder: &mut ProductionDecoder<'_>,
    tokenizer: &SubwordTokenizer,
    prompt: &OpenGenerationPrompt,
) -> Result<PromptScore, Box<dyn std::error::Error>> {
    let tokens = tokenizer.encode(&prompt.prompt);
    if tokens.is_empty() {
        return Err(format!("prompt {} encodes to no candidate tokens", prompt.id).into());
    }
    decoder.reset();
    let mut forward = decoder.step(BOS_TOKEN_ID)?;
    let mut total_nll_millibits = 0_u64;
    let mut zero_probability_targets = 0_usize;
    let mut residual_saturation_count = forward.residual_saturation_count;
    for (index, &target) in tokens.iter().enumerate() {
        let target = target as usize;
        let max_logit = forward
            .logits_q8
            .iter()
            .copied()
            .max()
            .ok_or("empty logits")?;
        zero_probability_targets = zero_probability_targets.saturating_add(usize::from(
            base2_exp_neg_q15(forward.logits_q8[target].saturating_sub(max_logit)) == 0,
        ));
        total_nll_millibits = total_nll_millibits
            .checked_add(
                base2_softmax_nll_millibits(
                    &forward.logits_q8,
                    target,
                    DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
                )
                .ok_or("canonical modeling NLL failed")?,
            )
            .ok_or("prompt modeling NLL overflow")?;
        if index + 1 < tokens.len() {
            forward = decoder.step(tokens[index])?;
            residual_saturation_count =
                residual_saturation_count.saturating_add(forward.residual_saturation_count);
        }
    }
    Ok(PromptScore {
        id: prompt.id.clone(),
        original_utf8_bytes: prompt.prompt.len(),
        candidate_tokens: tokens.len(),
        total_nll_millibits,
        zero_probability_targets,
        residual_saturation_count,
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut manifest = PathBuf::from("benchmarks/open-generation-v1/manifest.tsv");
    let mut tokenizer = None;
    let mut model = None;
    let mut trace = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
            args.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--manifest" => manifest = PathBuf::from(value(&mut args, "--manifest")?),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value(&mut args, "--tokenizer")?)),
            "--model" => model = Some(PathBuf::from(value(&mut args, "--model")?)),
            "--trace" => trace = Some(PathBuf::from(value(&mut args, "--trace")?)),
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {arg}\n{}", usage())),
        }
    }
    Ok(Config {
        manifest,
        tokenizer: tokenizer.ok_or_else(|| "--tokenizer is required".to_string())?,
        model: model.ok_or_else(|| "--model is required".to_string())?,
        trace: trace.ok_or_else(|| "--trace is required".to_string())?,
    })
}

fn usage() -> String {
    "Usage: nsrl-open-generation-modeling-run --manifest PATH --tokenizer PATH --model PATH --trace PATH".to_string()
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use nsrl_corpus::subword::SubwordTrainConfig;
    use nsrl_train::production::ProductionModelConfig;

    use super::*;

    #[test]
    fn modeling_score_is_deterministic_and_resets_the_decoder() {
        let (tokenizer, _) = SubwordTokenizer::train(
            b"integer language integer language integer language",
            SubwordTrainConfig {
                target_vocab_size: 320,
                min_pair_frequency: 2,
            },
        )
        .expect("tokenizer");
        let model = ProductionModelV1::new_initial(
            ProductionModelConfig {
                vocab_size: 320,
                d_model: 16,
                heads: 4,
                layers: 2,
                hidden_dim: 48,
                context_tokens: 16,
            },
            tokenizer.tokenizer_hash(),
            7,
        )
        .expect("model");
        let prompt = OpenGenerationPrompt {
            id: "test-prompt".to_string(),
            category: "continuation".to_string(),
            max_new_tokens: 8,
            required_phrase: Vec::new(),
            prompt: b"integer language".to_vec(),
        };
        let mut decoder = ProductionDecoder::new(&model).expect("decoder");
        let left = score_prompt(&mut decoder, &tokenizer, &prompt).expect("left score");
        let right = score_prompt(&mut decoder, &tokenizer, &prompt).expect("right score");
        assert_eq!(left.original_utf8_bytes, prompt.prompt.len());
        assert_eq!(
            left.candidate_tokens,
            tokenizer.encode(&prompt.prompt).len()
        );
        assert!(left.total_nll_millibits > 0);
        assert_eq!(left.total_nll_millibits, right.total_nll_millibits);
        assert_eq!(
            left.residual_saturation_count,
            right.residual_saturation_count
        );
    }
}
