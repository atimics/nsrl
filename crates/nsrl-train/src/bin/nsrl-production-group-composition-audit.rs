#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_train::production::{
    ProductionModelV1, decode_bound_token_stream,
    evaluate_production_model_canonical_nll_default_floor,
};

const SCHEMA: &str = "nsrl.production_group_composition_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-group-composition-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const GROUPS: [&str; 4] = ["embeddings", "k", "v", "o"];

#[derive(Debug)]
struct Config {
    tokenizer: PathBuf,
    tokens: PathBuf,
    source: PathBuf,
    candidate: PathBuf,
    trace: PathBuf,
    context_tokens: usize,
    max_windows: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-group-composition-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let token_bytes = fs::read(&config.tokens)?;
    let source_bytes = fs::read(&config.source)?;
    let candidate_bytes = fs::read(&config.candidate)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let source = ProductionModelV1::from_bytes(&source_bytes)?;
    let candidate = ProductionModelV1::from_bytes(&candidate_bytes)?;
    if tokenizer.tokenizer_hash() != source.tokenizer_hash
        || tokenizer.tokenizer_hash() != candidate.tokenizer_hash
        || source.config != candidate.config
        || source.scales != candidate.scales
        || source.initialization_seed != candidate.initialization_seed
    {
        return Err("source, candidate, and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &token_bytes,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let recomposed = compose(&source, &candidate, &GROUPS)?;
    if recomposed != candidate {
        return Err("candidate changes parameters outside embeddings, K, V, and O".into());
    }

    let source_eval = evaluate_production_model_canonical_nll_default_floor(
        &source,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    let subsets = [
        ("source", Vec::new()),
        ("embeddings_only", vec!["embeddings"]),
        ("k_only", vec!["k"]),
        ("v_only", vec!["v"]),
        ("o_only", vec!["o"]),
        ("without_embeddings", vec!["k", "v", "o"]),
        ("without_k", vec!["embeddings", "v", "o"]),
        ("without_v", vec!["embeddings", "k", "o"]),
        ("without_o", vec!["embeddings", "k", "v"]),
        ("candidate", GROUPS.to_vec()),
    ];
    let mut rows = String::new();
    for (index, (id, groups)) in subsets.iter().enumerate() {
        let model = if groups.is_empty() {
            source.clone()
        } else if groups.len() == GROUPS.len() {
            candidate.clone()
        } else {
            compose(&source, &candidate, groups)?
        };
        let evaluation = evaluate_production_model_canonical_nll_default_floor(
            &model,
            &tokens,
            token_stream_hash,
            config.context_tokens,
            config.max_windows,
        )?;
        if index > 0 {
            rows.push(',');
        }
        let rendered_groups = groups
            .iter()
            .map(|group| format!("\"{group}\""))
            .collect::<Vec<_>>()
            .join(",");
        write!(
            rows,
            concat!(
                "{{\"id\":\"{}\",\"candidate_groups\":[{}],",
                "\"model_hash\":\"0x{:016x}\",",
                "\"total_nll_millibits\":{},\"delta_from_source_millibits\":{},",
                "\"mistakes\":{},\"zero_probability_windows\":{},",
                "\"residual_saturation_count\":{}}}"
            ),
            id,
            rendered_groups,
            evaluation.model_hash,
            evaluation.total_nll_millibits,
            signed_delta(
                evaluation.total_nll_millibits,
                source_eval.total_nll_millibits,
            ),
            evaluation.mistakes,
            evaluation.zero_probability_windows,
            evaluation.residual_saturation_count,
        )?;
    }

    let output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"source_model_hash\":\"0x{:016x}\",",
            "\"candidate_model_hash\":\"0x{:016x}\",",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"token_stream_hash\":\"0x{:016x}\",",
            "\"source_artifact_fnv64\":\"0x{:016x}\",",
            "\"candidate_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"token_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"evaluation\":{{\"partition\":\"development\",",
            "\"metric\":\"canonical_integer_base2_softmax_nll_millibits\",",
            "\"context_tokens\":{},\"windows\":{}}},",
            "\"groups\":[\"embeddings\",\"k\",\"v\",\"o\"],",
            "\"candidate_diff_isolated_to_groups\":true,",
            "\"rows\":[{}]}}\n"
        ),
        SCHEMA,
        source.model_hash(),
        candidate.model_hash(),
        source.tokenizer_hash,
        token_stream_hash,
        fnv64(&source_bytes),
        fnv64(&candidate_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&token_bytes),
        fnv64(AUDIT_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        config.context_tokens,
        source_eval.windows,
        rows,
    );
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn compose(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    groups: &[&str],
) -> Result<ProductionModelV1, Box<dyn std::error::Error>> {
    let mut composed = source.clone();
    for group in groups {
        match *group {
            "embeddings" => composed.embeddings.clone_from(&candidate.embeddings),
            "k" => composed.k_weights.clone_from(&candidate.k_weights),
            "v" => composed.v_weights.clone_from(&candidate.v_weights),
            "o" => composed.o_weights.clone_from(&candidate.o_weights),
            _ => return Err(format!("unsupported composition group: {group}").into()),
        }
    }
    composed.validate()?;
    Ok(composed)
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut tokenizer = None;
    let mut tokens = None;
    let mut source = None;
    let mut candidate = None;
    let mut trace = None;
    let mut context_tokens = 64_usize;
    let mut max_windows = 512_usize;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = || "missing group composition audit argument value";
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--tokens" => tokens = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--source" => source = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--candidate" => candidate = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--trace" => trace = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--context-tokens" => context_tokens = args.next().ok_or_else(value)?.parse()?,
            "--max-windows" => max_windows = args.next().ok_or_else(value)?.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        source: source.ok_or("--source is required")?,
        candidate: candidate.ok_or("--candidate is required")?,
        trace: trace.ok_or("--trace is required")?,
        context_tokens,
        max_windows,
    })
}

fn signed_delta(value: u64, source: u64) -> i64 {
    if value >= source {
        i64::try_from(value - source).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(source - value).unwrap_or(i64::MAX)
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::signed_delta;

    #[test]
    fn signed_delta_preserves_direction() {
        assert_eq!(signed_delta(12, 10), 2);
        assert_eq!(signed_delta(8, 10), -2);
    }
}
