#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_eval::open_generation::{
    OPEN_GENERATION_CONTRACT_ID, load_open_generation_development_panel,
    load_open_generation_manifest,
};
use nsrl_train::production::{
    ProductionGenerationConfig, ProductionModelV1, generate_production_model,
};

const SAMPLE_SCHEMA: &str = "nsrl.open_generation_sample.v1";
const RUN_SCHEMA: &str = "nsrl.open_generation_run.v1";
const RUNNER_SOURCE: &[u8] = include_bytes!("nsrl-open-generation-run.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    manifest: PathBuf,
    tokenizer: PathBuf,
    model: PathBuf,
    samples_out: PathBuf,
    decoder_traces_out: PathBuf,
    trace: PathBuf,
    top_k: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-open-generation-run: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let runner_binary_hash = fnv64(&fs::read(env::current_exe()?)?);
    let manifest = load_open_generation_manifest(&config.manifest)?;
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
    if config.top_k < 2 || config.top_k > model.config.vocab_size.saturating_sub(2) {
        return Err("--top-k must be in 2..candidate vocabulary minus special tokens".into());
    }
    let model_hash = model.model_hash();

    let expected_samples = prompts.len() * (manifest.sampling_seeds.len() + 1);
    let mut samples = String::new();
    let mut decoder_traces = String::new();
    let mut total_generated_tokens = 0_usize;
    let mut residual_saturation_count = 0_usize;
    let mut maximum_cache_state_bytes = 0_usize;
    let mut maximum_cache_workspace_bytes = 0_usize;
    let mut maximum_cache_tokens = 0_usize;
    let mut samples_beyond_training_context = 0_usize;

    for prompt in &prompts {
        let prompt_tokens = tokenizer.encode(&prompt.prompt);
        if prompt_tokens.is_empty() || prompt_tokens.len() > model.config.context_tokens {
            return Err(format!(
                "prompt {} encodes to {} tokens outside candidate context 1..={} ",
                prompt.id,
                prompt_tokens.len(),
                model.config.context_tokens,
            )
            .into());
        }
        let modes = core::iter::once(("greedy", 0_u64, 1_usize)).chain(
            manifest
                .sampling_seeds
                .iter()
                .copied()
                .map(|seed| ("sample", seed, config.top_k)),
        );
        for (mode, seed, top_k) in modes {
            let generation = generate_production_model(
                &model,
                &prompt_tokens,
                ProductionGenerationConfig {
                    context_tokens: model.config.context_tokens,
                    max_new_tokens: manifest.generation_tokens,
                    top_k,
                    sample_seed: seed,
                    stop_on_eos: false,
                },
            )?;
            if generation.generated_tokens.len() != manifest.generation_tokens {
                return Err(format!("{} {mode} generation was truncated", prompt.id).into());
            }
            let generated = tokenizer.decode(&generation.generated_tokens)?;
            push_sample_json(
                &mut samples,
                model_hash,
                model.tokenizer_hash,
                prompt,
                mode,
                seed,
                top_k,
                prompt_tokens.len(),
                &generation.generated_tokens,
                &generated,
                generation.generated_token_hash,
                generation.residual_saturation_count,
                generation.cache_state_bytes,
                generation.cache_workspace_bytes,
                generation.cache_tokens_processed,
                generation.steps_beyond_training_context,
            );
            decoder_traces.push_str(&generation.to_json_line());
            total_generated_tokens =
                total_generated_tokens.saturating_add(generation.generated_tokens.len());
            residual_saturation_count =
                residual_saturation_count.saturating_add(generation.residual_saturation_count);
            maximum_cache_state_bytes = maximum_cache_state_bytes.max(generation.cache_state_bytes);
            maximum_cache_workspace_bytes =
                maximum_cache_workspace_bytes.max(generation.cache_workspace_bytes);
            maximum_cache_tokens = maximum_cache_tokens.max(generation.cache_tokens_processed);
            samples_beyond_training_context = samples_beyond_training_context
                .saturating_add(usize::from(generation.steps_beyond_training_context > 0));
        }
    }
    if samples.lines().count() != expected_samples
        || decoder_traces.lines().count() != expected_samples
    {
        return Err("open-generation runner emitted the wrong sample count".into());
    }

    write_file(&config.samples_out, samples.as_bytes())?;
    write_file(&config.decoder_traces_out, decoder_traces.as_bytes())?;
    let trace = format!(
        concat!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",",
            "\"partition\":\"development\",\"execution\":",
            "\"incremental_linear_attention_cache_v1\",",
            "\"bindings\":{{\"manifest_fnv64\":\"0x{:016x}\",",
            "\"contract_tokenizer_fnv64\":\"{}\",",
            "\"candidate_tokenizer_fnv64\":\"0x{:016x}\",",
            "\"candidate_model_fnv64\":\"0x{:016x}\",",
            "\"runner_source_fnv64\":\"0x{:016x}\",",
            "\"runner_binary_fnv64\":\"0x{:016x}\",",
            "\"samples_fnv64\":\"0x{:016x}\",",
            "\"decoder_traces_fnv64\":\"0x{:016x}\"}},",
            "\"sampling\":{{\"greedy\":true,\"top_k\":{},\"seeds\":[{}]}},",
            "\"counts\":{{\"prompts\":{},\"samples\":{},",
            "\"generated_tokens\":{},\"samples_beyond_training_context\":{}}},",
            "\"cache\":{{\"maximum_state_bytes\":{},\"maximum_workspace_bytes\":{},",
            "\"maximum_tokens_processed\":{}}},",
            "\"residual_saturation_count\":{},",
            "\"forbidden_assistance\":{{\"retrieval\":false,\"corpus_prior\":false,",
            "\"memory_injection\":false,\"target_lookup\":false,",
            "\"routing_oracle\":false}}}}\n"
        ),
        RUN_SCHEMA,
        OPEN_GENERATION_CONTRACT_ID,
        fnv64(&fs::read(&config.manifest)?),
        manifest.tokenizer_hash,
        tokenizer.tokenizer_hash(),
        model_hash,
        fnv64(RUNNER_SOURCE),
        runner_binary_hash,
        fnv64(samples.as_bytes()),
        fnv64(decoder_traces.as_bytes()),
        config.top_k,
        manifest
            .sampling_seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
        prompts.len(),
        expected_samples,
        total_generated_tokens,
        samples_beyond_training_context,
        maximum_cache_state_bytes,
        maximum_cache_workspace_bytes,
        maximum_cache_tokens,
        residual_saturation_count,
    );
    write_file(&config.trace, trace.as_bytes())?;
    print!("{trace}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_sample_json(
    output: &mut String,
    model_hash: u64,
    tokenizer_hash: u64,
    prompt: &nsrl_eval::open_generation::OpenGenerationPrompt,
    mode: &str,
    seed: u64,
    top_k: usize,
    prompt_token_count: usize,
    generated_tokens: &[u32],
    generated: &[u8],
    generated_token_hash: u64,
    residual_saturation_count: usize,
    cache_state_bytes: usize,
    cache_workspace_bytes: usize,
    cache_tokens_processed: usize,
    steps_beyond_training_context: usize,
) {
    write!(
        output,
        concat!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",",
            "\"partition\":\"development\",\"prompt_id\":\"{}\",",
            "\"category\":\"{}\",\"mode\":\"{}\",\"seed\":{},",
            "\"top_k\":{},\"bindings\":{{\"model_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_fnv64\":\"0x{:016x}\"}},",
            "\"prompt_token_count\":{},\"required_phrase_hex\":\"{}\",",
            "\"generated_token_count\":{},",
            "\"generated_token_fnv64\":\"0x{:016x}\",\"generated_tokens\":["
        ),
        SAMPLE_SCHEMA,
        OPEN_GENERATION_CONTRACT_ID,
        prompt.id,
        prompt.category,
        mode,
        seed,
        top_k,
        model_hash,
        tokenizer_hash,
        prompt_token_count,
        hex(&prompt.required_phrase),
        generated_tokens.len(),
        generated_token_hash,
    )
    .expect("writing sample JSON cannot fail");
    for (index, token) in generated_tokens.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{token}").expect("writing token cannot fail");
    }
    write!(
        output,
        concat!(
            "],\"generated_hex\":\"{}\",",
            "\"execution\":{{\"decoder\":\"incremental_linear_attention_cache_v1\",",
            "\"residual_saturation_count\":{},\"cache_state_bytes\":{},",
            "\"cache_workspace_bytes\":{},\"cache_tokens_processed\":{},",
            "\"steps_beyond_training_context\":{}}}}}\n"
        ),
        hex(generated),
        residual_saturation_count,
        cache_state_bytes,
        cache_workspace_bytes,
        cache_tokens_processed,
        steps_beyond_training_context,
    )
    .expect("writing sample JSON tail cannot fail");
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut manifest = PathBuf::from("benchmarks/open-generation-v1/manifest.tsv");
    let mut tokenizer = None;
    let mut model = None;
    let mut samples_out = None;
    let mut decoder_traces_out = None;
    let mut trace = None;
    let mut top_k = 40_usize;
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
            "--samples-out" => {
                samples_out = Some(PathBuf::from(value(&mut args, "--samples-out")?))
            }
            "--decoder-traces-out" => {
                decoder_traces_out = Some(PathBuf::from(value(&mut args, "--decoder-traces-out")?))
            }
            "--trace" => trace = Some(PathBuf::from(value(&mut args, "--trace")?)),
            "--top-k" => {
                top_k = value(&mut args, "--top-k")?
                    .parse()
                    .map_err(|_| "--top-k must be an integer".to_string())?
            }
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {arg}\n{}", usage())),
        }
    }
    Ok(Config {
        manifest,
        tokenizer: tokenizer.ok_or_else(|| "--tokenizer is required".to_string())?,
        model: model.ok_or_else(|| "--model is required".to_string())?,
        samples_out: samples_out.ok_or_else(|| "--samples-out is required".to_string())?,
        decoder_traces_out: decoder_traces_out
            .ok_or_else(|| "--decoder-traces-out is required".to_string())?,
        trace: trace.ok_or_else(|| "--trace is required".to_string())?,
        top_k,
    })
}

fn usage() -> String {
    "Usage: nsrl-open-generation-run --manifest PATH --tokenizer PATH --model PATH --samples-out PATH --decoder-traces-out PATH --trace PATH [--top-k N]".to_string()
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing hex cannot fail");
    }
    output
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}
