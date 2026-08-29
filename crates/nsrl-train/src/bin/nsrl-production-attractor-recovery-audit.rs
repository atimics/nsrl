#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, SubwordTokenizer};
use nsrl_eval::open_generation::{
    load_open_generation_development_panel, load_open_generation_manifest,
};
use nsrl_train::TrainError;
use nsrl_train::production::{
    ProductionDecoder, ProductionForward, ProductionModelV1, decode_bound_token_stream,
    forward_production_model,
};

const SCHEMA: &str = "nsrl.production_attractor_recovery_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-attractor-recovery-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    manifest: PathBuf,
    tokenizer: PathBuf,
    tokens: PathBuf,
    source_model: PathBuf,
    candidate_model: PathBuf,
    trace: PathBuf,
    context_tokens: usize,
    rollout_tokens: usize,
    max_windows: usize,
}

#[derive(Debug)]
struct TokenizedPrompt {
    id: String,
    tokens: Vec<u32>,
}

#[derive(Debug)]
struct PromptRow {
    id: String,
    prompt_tokens: usize,
    cache_exact: bool,
    greedy_token: usize,
    feedback_token: usize,
    immediate_fixed_point: bool,
    isolated_token: usize,
    isolated_fixed_point: bool,
    reversed_feature_changed: bool,
    reversed_logits_changed: bool,
    reversed_argmax_changed: bool,
    residual_saturation_count: usize,
}

#[derive(Debug)]
struct PromptAudit {
    rows: Vec<PromptRow>,
    features: Vec<Vec<i16>>,
}

#[derive(Debug, Default)]
struct RolloutStats {
    reference_matches: usize,
    self_loop_transitions: usize,
    two_cycle_transitions: usize,
    minimum_unique_tokens: usize,
    maximum_unique_tokens: usize,
    residual_saturation_count: usize,
}

#[derive(Debug)]
struct ModelAudit {
    model_hash: u64,
    prompt: PromptAudit,
    ordinary: RolloutStats,
    loop_masked: RolloutStats,
}

#[derive(Debug)]
struct ContinuationWindow {
    context: Vec<u32>,
    continuation: Vec<u32>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-attractor-recovery-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let manifest_bytes = fs::read(&config.manifest)?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let token_bytes = fs::read(&config.tokens)?;
    let source_bytes = fs::read(&config.source_model)?;
    let candidate_bytes = fs::read(&config.candidate_model)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let source = ProductionModelV1::from_bytes(&source_bytes)?;
    let candidate = ProductionModelV1::from_bytes(&candidate_bytes)?;
    if source.config != candidate.config
        || source.tokenizer_hash != candidate.tokenizer_hash
        || source.tokenizer_hash != tokenizer.tokenizer_hash()
        || source.config.vocab_size != tokenizer.vocab_size()
        || config.context_tokens < 2
        || config.context_tokens > source.config.context_tokens
        || config.rollout_tokens < 3
        || config.max_windows < 2
    {
        return Err("audit configuration or model binding is invalid".into());
    }

    let manifest = load_open_generation_manifest(&config.manifest)?;
    let prompts = load_open_generation_development_panel(&manifest)?
        .into_iter()
        .map(|prompt| TokenizedPrompt {
            id: prompt.id,
            tokens: tokenizer.encode(&prompt.prompt),
        })
        .collect::<Vec<_>>();
    if prompts.iter().any(|prompt| {
        prompt.tokens.is_empty() || prompt.tokens.len() > source.config.context_tokens
    }) {
        return Err("development prompt is outside the model context".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &token_bytes,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let windows = spread_continuation_windows(
        &tokens,
        config.context_tokens,
        config.rollout_tokens,
        config.max_windows,
    );
    if windows.len() != config.max_windows {
        return Err("development token stream has too few continuation windows".into());
    }

    let source_audit = audit_model(&source, &prompts, &windows)?;
    let candidate_audit = audit_model(&candidate, &prompts, &windows)?;
    let cross_model_feature_mismatches = source_audit
        .prompt
        .features
        .iter()
        .zip(&candidate_audit.prompt.features)
        .filter(|(left, right)| left != right)
        .count();
    let positions = windows.len() * config.rollout_tokens;
    let two_cycle_positions = windows.len() * config.rollout_tokens.saturating_sub(2);
    let candidate_cache_exact = candidate_audit
        .prompt
        .rows
        .iter()
        .all(|row| row.cache_exact);
    let candidate_masked_match_per_mille =
        per_mille(candidate_audit.loop_masked.reference_matches, positions);
    let candidate_masked_two_cycle_per_mille = per_mille(
        candidate_audit.loop_masked.two_cycle_transitions,
        two_cycle_positions,
    );
    let gates = [
        candidate_cache_exact,
        cross_model_feature_mismatches == 0,
        candidate_masked_match_per_mille >= 50,
        candidate_audit.loop_masked.minimum_unique_tokens >= 8,
        candidate_masked_two_cycle_per_mille <= 250,
        candidate_audit.loop_masked.residual_saturation_count == 0,
    ];
    let training_candidate_justified = gates.into_iter().all(|gate| gate);

    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"source_model_hash\":\"0x{:016x}\",",
            "\"candidate_model_hash\":\"0x{:016x}\",",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"token_stream_hash\":\"0x{:016x}\",",
            "\"source_model_artifact_fnv64\":\"0x{:016x}\",",
            "\"candidate_model_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"token_artifact_fnv64\":\"0x{:016x}\",",
            "\"manifest_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"prompts\":{},\"windows\":{},",
            "\"context_tokens\":{},\"rollout_tokens\":{},",
            "\"evaluated_positions\":{}}},",
            "\"cross_model\":{{\"prompt_feature_mismatches\":{}}},",
            "\"models\":{{\"source\":"
        ),
        SCHEMA,
        source.model_hash(),
        candidate.model_hash(),
        tokenizer.tokenizer_hash(),
        token_stream_hash,
        fnv64(&source_bytes),
        fnv64(&candidate_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&token_bytes),
        fnv64(&manifest_bytes),
        fnv64(AUDIT_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        prompts.len(),
        windows.len(),
        config.context_tokens,
        config.rollout_tokens,
        positions,
        cross_model_feature_mismatches,
    );
    write_model_audit(&mut output, &source_audit, positions, two_cycle_positions)?;
    output.push_str(",\"candidate\":");
    write_model_audit(
        &mut output,
        &candidate_audit,
        positions,
        two_cycle_positions,
    )?;
    write!(
        output,
        concat!(
            "}},\"decision\":{{",
            "\"candidate_cache_exact\":{},",
            "\"cross_model_prompt_features_exact\":{},",
            "\"loop_masked_reference_match_per_mille\":{},",
            "\"loop_masked_reference_match_minimum\":50,",
            "\"loop_masked_minimum_unique_tokens\":{},",
            "\"loop_masked_unique_tokens_minimum\":8,",
            "\"loop_masked_two_cycle_per_mille\":{},",
            "\"loop_masked_two_cycle_maximum\":250,",
            "\"loop_masked_zero_residual_saturation\":{},",
            "\"training_candidate_justified\":{}}},",
            "\"hidden_panel_opened\":false}}\n"
        ),
        candidate_cache_exact,
        cross_model_feature_mismatches == 0,
        candidate_masked_match_per_mille,
        candidate_audit.loop_masked.minimum_unique_tokens,
        candidate_masked_two_cycle_per_mille,
        candidate_audit.loop_masked.residual_saturation_count == 0,
        training_candidate_justified,
    )?;
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn audit_model(
    model: &ProductionModelV1,
    prompts: &[TokenizedPrompt],
    windows: &[ContinuationWindow],
) -> Result<ModelAudit, TrainError> {
    Ok(ModelAudit {
        model_hash: model.model_hash(),
        prompt: audit_prompts(model, prompts)?,
        ordinary: audit_rollouts(model, windows, false)?,
        loop_masked: audit_rollouts(model, windows, true)?,
    })
}

fn audit_prompts(
    model: &ProductionModelV1,
    prompts: &[TokenizedPrompt],
) -> Result<PromptAudit, TrainError> {
    let mut rows = Vec::with_capacity(prompts.len());
    let mut features = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let (mut decoder, incremental, mut residual_saturation_count) =
            prime_decoder(model, &prompt.tokens)?;
        let full = forward_production_model(model, &prompt.tokens)?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(full.residual_saturation_count);
        let greedy_token = select_token(&incremental.logits_q8, None);
        let feedback = decoder.step(greedy_token as u32)?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(feedback.residual_saturation_count);
        let feedback_token = select_token(&feedback.logits_q8, None);
        let isolated = forward_production_model(model, &[greedy_token as u32])?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(isolated.residual_saturation_count);
        let isolated_token = select_token(&isolated.logits_q8, None);
        let mut reversed_tokens = prompt.tokens.clone();
        let prefix_end = reversed_tokens.len().saturating_sub(1);
        reversed_tokens[..prefix_end].reverse();
        let reversed = forward_production_model(model, &reversed_tokens)?;
        residual_saturation_count =
            residual_saturation_count.saturating_add(reversed.residual_saturation_count);
        features.push(full.features_q15.clone());
        rows.push(PromptRow {
            id: prompt.id.clone(),
            prompt_tokens: prompt.tokens.len(),
            cache_exact: incremental == full,
            greedy_token,
            feedback_token,
            immediate_fixed_point: feedback_token == greedy_token,
            isolated_token,
            isolated_fixed_point: isolated_token == greedy_token,
            reversed_feature_changed: reversed.features_q15 != full.features_q15,
            reversed_logits_changed: reversed.logits_q8 != full.logits_q8,
            reversed_argmax_changed: select_token(&reversed.logits_q8, None) != greedy_token,
            residual_saturation_count,
        });
    }
    Ok(PromptAudit { rows, features })
}

fn audit_rollouts(
    model: &ProductionModelV1,
    windows: &[ContinuationWindow],
    mask_previous: bool,
) -> Result<RolloutStats, TrainError> {
    let mut stats = RolloutStats {
        minimum_unique_tokens: usize::MAX,
        ..RolloutStats::default()
    };
    for window in windows {
        let (mut decoder, mut forward, saturation) = prime_decoder(model, &window.context)?;
        stats.residual_saturation_count =
            stats.residual_saturation_count.saturating_add(saturation);
        let mut previous = window
            .context
            .last()
            .copied()
            .ok_or(TrainError::InvalidConfig)?;
        let mut generated = Vec::with_capacity(window.continuation.len());
        for (step, &target) in window.continuation.iter().enumerate() {
            let excluded = mask_previous.then_some(previous as usize);
            let selected = select_token(&forward.logits_q8, excluded) as u32;
            stats.reference_matches = stats
                .reference_matches
                .saturating_add(usize::from(selected == target));
            generated.push(selected);
            previous = selected;
            if step + 1 < window.continuation.len() {
                forward = decoder.step(selected)?;
                stats.residual_saturation_count = stats
                    .residual_saturation_count
                    .saturating_add(forward.residual_saturation_count);
            }
        }
        let unique = unique_count(&generated);
        stats.minimum_unique_tokens = stats.minimum_unique_tokens.min(unique);
        stats.maximum_unique_tokens = stats.maximum_unique_tokens.max(unique);
        stats.self_loop_transitions = stats.self_loop_transitions.saturating_add(
            generated
                .windows(2)
                .filter(|pair| pair[0] == pair[1])
                .count(),
        );
        stats.two_cycle_transitions = stats
            .two_cycle_transitions
            .saturating_add(two_cycle_count(&generated));
    }
    if windows.is_empty() {
        stats.minimum_unique_tokens = 0;
    }
    Ok(stats)
}

fn write_model_audit(
    output: &mut String,
    audit: &ModelAudit,
    positions: usize,
    two_cycle_positions: usize,
) -> std::fmt::Result {
    let prompt_unique_greedy_tokens = unique_count(
        &audit
            .prompt
            .rows
            .iter()
            .map(|row| row.greedy_token as u32)
            .collect::<Vec<_>>(),
    );
    write!(
        output,
        concat!(
            "{{\"model_hash\":\"0x{:016x}\",\"prompt\":{{",
            "\"cache_mismatches\":{},\"unique_greedy_tokens\":{},",
            "\"immediate_fixed_points\":{},\"isolated_fixed_points\":{},",
            "\"reversed_prefix_feature_changes\":{},",
            "\"reversed_prefix_logit_changes\":{},",
            "\"reversed_prefix_argmax_changes\":{},",
            "\"residual_saturation_count\":{},\"rows\":["
        ),
        audit.model_hash,
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| !row.cache_exact)
            .count(),
        prompt_unique_greedy_tokens,
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| row.immediate_fixed_point)
            .count(),
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| row.isolated_fixed_point)
            .count(),
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| row.reversed_feature_changed)
            .count(),
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| row.reversed_logits_changed)
            .count(),
        audit
            .prompt
            .rows
            .iter()
            .filter(|row| row.reversed_argmax_changed)
            .count(),
        audit
            .prompt
            .rows
            .iter()
            .map(|row| row.residual_saturation_count)
            .sum::<usize>(),
    )?;
    for (index, row) in audit.prompt.rows.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            concat!(
                "{{\"id\":\"{}\",\"prompt_tokens\":{},\"cache_exact\":{},",
                "\"greedy_token\":{},\"feedback_token\":{},",
                "\"immediate_fixed_point\":{},\"isolated_token\":{},",
                "\"isolated_fixed_point\":{},",
                "\"reversed_feature_changed\":{},",
                "\"reversed_logits_changed\":{},",
                "\"reversed_argmax_changed\":{},",
                "\"residual_saturation_count\":{}}}"
            ),
            row.id,
            row.prompt_tokens,
            row.cache_exact,
            row.greedy_token,
            row.feedback_token,
            row.immediate_fixed_point,
            row.isolated_token,
            row.isolated_fixed_point,
            row.reversed_feature_changed,
            row.reversed_logits_changed,
            row.reversed_argmax_changed,
            row.residual_saturation_count,
        )?;
    }
    output.push_str("],\"ordinary_rollout\":");
    write_rollout(output, &audit.ordinary, positions, two_cycle_positions)?;
    output.push_str(",\"loop_masked_rollout\":");
    write_rollout(output, &audit.loop_masked, positions, two_cycle_positions)?;
    output.push_str("}}");
    Ok(())
}

fn write_rollout(
    output: &mut String,
    stats: &RolloutStats,
    positions: usize,
    two_cycle_positions: usize,
) -> std::fmt::Result {
    write!(
        output,
        concat!(
            "{{\"reference_matches\":{},\"reference_match_per_mille\":{},",
            "\"self_loop_transitions\":{},",
            "\"minimum_unique_tokens\":{},\"maximum_unique_tokens\":{},",
            "\"two_cycle_transitions\":{},\"two_cycle_per_mille\":{},",
            "\"residual_saturation_count\":{}}}"
        ),
        stats.reference_matches,
        per_mille(stats.reference_matches, positions),
        stats.self_loop_transitions,
        stats.minimum_unique_tokens,
        stats.maximum_unique_tokens,
        stats.two_cycle_transitions,
        per_mille(stats.two_cycle_transitions, two_cycle_positions),
        stats.residual_saturation_count,
    )
}

fn prime_decoder<'model>(
    model: &'model ProductionModelV1,
    context: &[u32],
) -> Result<(ProductionDecoder<'model>, ProductionForward, usize), TrainError> {
    let mut decoder = ProductionDecoder::new(model)?;
    let mut forward = None;
    let mut saturation = 0_usize;
    for &token in context {
        let row = decoder.step(token)?;
        saturation = saturation.saturating_add(row.residual_saturation_count);
        forward = Some(row);
    }
    Ok((
        decoder,
        forward.ok_or(TrainError::InvalidConfig)?,
        saturation,
    ))
}

fn select_token(logits: &[i32], excluded: Option<usize>) -> usize {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, _)| !is_masked(*token) && Some(*token) != excluded)
        .max_by(|(left_token, left), (right_token, right)| {
            left.cmp(right).then_with(|| right_token.cmp(left_token))
        })
        .map(|(token, _)| token)
        .unwrap_or(0)
}

fn spread_continuation_windows(
    tokens: &[u32],
    context_tokens: usize,
    rollout_tokens: usize,
    max_windows: usize,
) -> Vec<ContinuationWindow> {
    let required = context_tokens.saturating_add(rollout_tokens);
    let mut total_windows = 0_usize;
    let mut document_tokens = 0_usize;
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document_tokens = 0;
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active && document_tokens >= required {
                total_windows = total_windows
                    .saturating_add(document_tokens.saturating_sub(required).saturating_add(1));
            }
            document_tokens = 0;
            active = false;
        } else if active {
            document_tokens = document_tokens.saturating_add(1);
        }
    }
    let selected = max_windows.min(total_windows);
    if selected == 0 {
        return Vec::new();
    }
    let ranks = if selected == 1 {
        vec![total_windows / 2]
    } else {
        (0..selected)
            .map(|index| {
                ((index as u128) * ((total_windows - 1) as u128) / ((selected - 1) as u128))
                    as usize
            })
            .collect::<Vec<_>>()
    };
    let mut windows = Vec::with_capacity(selected);
    let mut rank_cursor = 0_usize;
    let mut population_rank = 0_usize;
    let mut document = Vec::new();
    active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active && document.len() >= required {
                for start in 0..=document.len() - required {
                    if rank_cursor < ranks.len() && population_rank == ranks[rank_cursor] {
                        windows.push(ContinuationWindow {
                            context: document[start..start + context_tokens].to_vec(),
                            continuation: document[start + context_tokens..start + required]
                                .to_vec(),
                        });
                        rank_cursor += 1;
                        if rank_cursor == ranks.len() {
                            return windows;
                        }
                    }
                    population_rank = population_rank.saturating_add(1);
                }
            }
            document.clear();
            active = false;
        } else if active {
            document.push(token);
        }
    }
    windows
}

fn is_masked(token: usize) -> bool {
    matches!(token as u32, BOS_TOKEN_ID | PAD_TOKEN_ID)
}

fn unique_count(tokens: &[u32]) -> usize {
    let mut tokens = tokens.to_vec();
    tokens.sort_unstable();
    tokens.dedup();
    tokens.len()
}

fn two_cycle_count(tokens: &[u32]) -> usize {
    tokens
        .windows(3)
        .filter(|window| window[0] == window[2])
        .count()
}

fn per_mille(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1000) / denominator
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut manifest = None;
    let mut tokenizer = None;
    let mut tokens = None;
    let mut source_model = None;
    let mut candidate_model = None;
    let mut trace = None;
    let mut context_tokens = 64_usize;
    let mut rollout_tokens = 16_usize;
    let mut max_windows = 8_usize;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => manifest = Some(next_path(&mut args, "--manifest")?),
            "--tokenizer" => tokenizer = Some(next_path(&mut args, "--tokenizer")?),
            "--tokens" => tokens = Some(next_path(&mut args, "--tokens")?),
            "--source-model" => source_model = Some(next_path(&mut args, "--source-model")?),
            "--candidate-model" => {
                candidate_model = Some(next_path(&mut args, "--candidate-model")?)
            }
            "--trace" => trace = Some(next_path(&mut args, "--trace")?),
            "--context-tokens" => context_tokens = next_usize(&mut args, "--context-tokens")?,
            "--rollout-tokens" => rollout_tokens = next_usize(&mut args, "--rollout-tokens")?,
            "--max-windows" => max_windows = next_usize(&mut args, "--max-windows")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(Config {
        manifest: manifest.ok_or("--manifest is required")?,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        source_model: source_model.ok_or("--source-model is required")?,
        candidate_model: candidate_model.ok_or("--candidate-model is required")?,
        trace: trace.ok_or("--trace is required")?,
        context_tokens,
        rollout_tokens,
        max_windows,
    })
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(
        args.next()
            .ok_or_else(|| format!("{option} requires a path"))?,
    ))
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(args
        .next()
        .ok_or_else(|| format!("{option} requires an integer"))?
        .parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_mask_selects_the_next_eligible_token() {
        let mut logits = vec![0; 300];
        logits[BOS_TOKEN_ID as usize] = 100;
        logits[PAD_TOKEN_ID as usize] = 99;
        logits[10] = 98;
        logits[11] = 97;

        assert_eq!(select_token(&logits, None), 10);
        assert_eq!(select_token(&logits, Some(10)), 11);
    }

    #[test]
    fn two_cycle_count_detects_alternating_attractors() {
        assert_eq!(two_cycle_count(&[1, 2, 1, 2, 1]), 3);
        assert_eq!(two_cycle_count(&[1, 2, 3, 4]), 0);
    }
}
