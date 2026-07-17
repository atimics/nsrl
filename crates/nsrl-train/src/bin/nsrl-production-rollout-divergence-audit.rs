#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, SubwordTokenizer};
use nsrl_train::TrainError;
use nsrl_train::production::{
    ProductionDecoder, ProductionForward, ProductionModelV1, decode_bound_token_stream,
};

const SCHEMA: &str = "nsrl.production_rollout_divergence_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-rollout-divergence-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    tokenizer: PathBuf,
    tokens: PathBuf,
    model: PathBuf,
    trace: PathBuf,
    context_tokens: Option<usize>,
    rollout_tokens: usize,
    max_windows: usize,
}

#[derive(Debug)]
struct ContinuationWindow {
    population_rank: usize,
    context: Vec<u32>,
    continuation: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct LogitComparison {
    differing_logits: usize,
    logit_l1_q8: u64,
    target_logit_delta_q8: i64,
    target_probability_delta_q15: i64,
    argmax_changed: bool,
}

#[derive(Debug)]
struct WindowAudit {
    population_rank: usize,
    first_target: usize,
    native_top: usize,
    prefix_swapped_top: usize,
    suffix_swapped_top: usize,
    first_target_rank: usize,
    first_target_probability_q15: i16,
    teacher_forced_matches: usize,
    teacher_forced_rank_sum: usize,
    teacher_forced_maximum_rank: usize,
    teacher_forced_target_probability_sum_q15: i64,
    free_running_matches: usize,
    free_running_exact_prefix_tokens: usize,
    free_running_unique_tokens: usize,
    free_running_self_loop_transitions: usize,
    prefix_swap: LogitComparison,
    suffix_swap: LogitComparison,
    prefix_swap_input_differences: usize,
    suffix_swap_input_differences: usize,
    reference_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_token_hash: u64,
    residual_saturation_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-rollout-divergence-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let token_bytes = fs::read(&config.tokens)?;
    let model_bytes = fs::read(&config.model)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let context_tokens = config.context_tokens.unwrap_or(model.config.context_tokens);
    if context_tokens < 2
        || context_tokens > model.config.context_tokens
        || config.rollout_tokens == 0
        || config.max_windows < 2
    {
        return Err("audit dimensions are invalid".into());
    }
    let (tokens, token_stream_hash) =
        decode_bound_token_stream(&token_bytes, model.tokenizer_hash, model.config.vocab_size)?;
    let windows = spread_continuation_windows(
        &tokens,
        context_tokens,
        config.rollout_tokens,
        config.max_windows,
    );
    if windows.len() < 2 {
        return Err("token stream has fewer than two eligible continuation windows".into());
    }

    let split = context_tokens / 2;
    let mut audits = Vec::with_capacity(windows.len());
    for index in 0..windows.len() {
        let window = &windows[index];
        let paired = &windows[(index + 1) % windows.len()];
        let mut prefix_swapped = paired.context[..split].to_vec();
        prefix_swapped.extend_from_slice(&window.context[split..]);
        let mut suffix_swapped = window.context[..split].to_vec();
        suffix_swapped.extend_from_slice(&paired.context[split..]);
        let prefix_swap_input_differences = window.context[..split]
            .iter()
            .zip(&paired.context[..split])
            .filter(|(left, right)| left != right)
            .count();
        let suffix_swap_input_differences = window.context[split..]
            .iter()
            .zip(&paired.context[split..])
            .filter(|(left, right)| left != right)
            .count();

        let (mut teacher, mut teacher_forward, mut residual_saturation_count) =
            prime_decoder(&model, &window.context)?;
        let (mut free, mut free_forward, free_prime_saturation) =
            prime_decoder(&model, &window.context)?;
        residual_saturation_count = residual_saturation_count.saturating_add(free_prime_saturation);
        let (_, prefix_forward, prefix_saturation) = prime_decoder(&model, &prefix_swapped)?;
        let (_, suffix_forward, suffix_saturation) = prime_decoder(&model, &suffix_swapped)?;
        residual_saturation_count = residual_saturation_count
            .saturating_add(prefix_saturation)
            .saturating_add(suffix_saturation);

        let first_target = window.continuation[0] as usize;
        let native_top = select_token(&teacher_forward.logits_q8);
        let prefix_swapped_top = select_token(&prefix_forward.logits_q8);
        let suffix_swapped_top = select_token(&suffix_forward.logits_q8);
        let first_target_rank = token_rank(&teacher_forward.logits_q8, first_target);
        let first_target_probability_q15 = teacher_forward.probabilities_q15[first_target];
        let prefix_swap = compare_logits(&teacher_forward, &prefix_forward, first_target);
        let suffix_swap = compare_logits(&teacher_forward, &suffix_forward, first_target);

        let mut teacher_forced_matches = 0_usize;
        let mut teacher_forced_rank_sum = 0_usize;
        let mut teacher_forced_maximum_rank = 0_usize;
        let mut teacher_forced_target_probability_sum_q15 = 0_i64;
        let mut free_running_matches = 0_usize;
        let mut exact_prefix_open = true;
        let mut free_running_exact_prefix_tokens = 0_usize;
        let mut generated_tokens = Vec::with_capacity(window.continuation.len());
        for (step, &target) in window.continuation.iter().enumerate() {
            let teacher_prediction = select_token(&teacher_forward.logits_q8) as u32;
            teacher_forced_matches =
                teacher_forced_matches.saturating_add(usize::from(teacher_prediction == target));
            let teacher_rank = token_rank(&teacher_forward.logits_q8, target as usize);
            teacher_forced_rank_sum = teacher_forced_rank_sum.saturating_add(teacher_rank);
            teacher_forced_maximum_rank = teacher_forced_maximum_rank.max(teacher_rank);
            teacher_forced_target_probability_sum_q15 = teacher_forced_target_probability_sum_q15
                .saturating_add(i64::from(
                    teacher_forward.probabilities_q15[target as usize],
                ));
            let generated = select_token(&free_forward.logits_q8) as u32;
            generated_tokens.push(generated);
            let matches = generated == target;
            free_running_matches = free_running_matches.saturating_add(usize::from(matches));
            if exact_prefix_open && matches {
                free_running_exact_prefix_tokens =
                    free_running_exact_prefix_tokens.saturating_add(1);
            } else {
                exact_prefix_open = false;
            }
            if step + 1 < window.continuation.len() {
                teacher_forward = teacher.step(target)?;
                free_forward = free.step(generated)?;
                residual_saturation_count = residual_saturation_count
                    .saturating_add(teacher_forward.residual_saturation_count)
                    .saturating_add(free_forward.residual_saturation_count);
            }
        }
        let free_running_unique_tokens = unique_count(&generated_tokens);
        let free_running_self_loop_transitions = generated_tokens
            .windows(2)
            .filter(|pair| pair[0] == pair[1])
            .count();
        audits.push(WindowAudit {
            population_rank: window.population_rank,
            first_target,
            native_top,
            prefix_swapped_top,
            suffix_swapped_top,
            first_target_rank,
            first_target_probability_q15,
            teacher_forced_matches,
            teacher_forced_rank_sum,
            teacher_forced_maximum_rank,
            teacher_forced_target_probability_sum_q15,
            free_running_matches,
            free_running_exact_prefix_tokens,
            free_running_unique_tokens,
            free_running_self_loop_transitions,
            prefix_swap,
            suffix_swap,
            prefix_swap_input_differences,
            suffix_swap_input_differences,
            reference_tokens: window.continuation.clone(),
            generated_token_hash: hash_u32(&generated_tokens),
            generated_tokens,
            residual_saturation_count,
        });
    }

    let positions = audits.len().saturating_mul(config.rollout_tokens);
    let transitions = audits
        .len()
        .saturating_mul(config.rollout_tokens.saturating_sub(1));
    let teacher_matches: usize = audits.iter().map(|row| row.teacher_forced_matches).sum();
    let teacher_rank_sum: usize = audits.iter().map(|row| row.teacher_forced_rank_sum).sum();
    let teacher_maximum_rank = audits
        .iter()
        .map(|row| row.teacher_forced_maximum_rank)
        .max()
        .unwrap_or(0);
    let teacher_probability_sum_q15: i64 = audits
        .iter()
        .map(|row| row.teacher_forced_target_probability_sum_q15)
        .sum();
    let free_matches: usize = audits.iter().map(|row| row.free_running_matches).sum();
    let exact_prefix_tokens: usize = audits
        .iter()
        .map(|row| row.free_running_exact_prefix_tokens)
        .sum();
    let nonzero_prefix_windows = audits
        .iter()
        .filter(|row| row.free_running_exact_prefix_tokens > 0)
        .count();
    let self_loop_transitions: usize = audits
        .iter()
        .map(|row| row.free_running_self_loop_transitions)
        .sum();
    let prefix_l1: u64 = audits.iter().map(|row| row.prefix_swap.logit_l1_q8).sum();
    let suffix_l1: u64 = audits.iter().map(|row| row.suffix_swap.logit_l1_q8).sum();
    let prefix_target_abs_delta: u64 = audits
        .iter()
        .map(|row| row.prefix_swap.target_logit_delta_q8.unsigned_abs())
        .sum();
    let suffix_target_abs_delta: u64 = audits
        .iter()
        .map(|row| row.suffix_swap.target_logit_delta_q8.unsigned_abs())
        .sum();
    let prefix_input_differences: usize = audits
        .iter()
        .map(|row| row.prefix_swap_input_differences)
        .sum();
    let suffix_input_differences: usize = audits
        .iter()
        .map(|row| row.suffix_swap_input_differences)
        .sum();
    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"model_hash\":\"0x{:016x}\",\"tokenizer_hash\":\"0x{:016x}\",",
            "\"token_stream_hash\":\"0x{:016x}\",\"model_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",\"token_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"counts\":{{\"windows\":{},\"context_tokens\":{},\"rollout_tokens\":{},",
            "\"evaluated_positions\":{},\"free_running_transitions\":{},\"context_split\":{}}},",
            "\"teacher_forced\":{{\"top1_matches\":{},\"top1_per_mille\":{},",
            "\"mean_target_rank\":{},\"maximum_target_rank\":{},",
            "\"mean_target_probability_q15\":{},\"mean_first_target_rank\":{},",
            "\"maximum_first_target_rank\":{}}},",
            "\"free_running\":{{\"reference_matches\":{},\"reference_match_per_mille\":{},",
            "\"exact_prefix_tokens\":{},\"windows_with_nonzero_exact_prefix\":{},",
            "\"self_loop_transitions\":{},\"self_loop_transition_per_mille\":{},",
            "\"minimum_unique_tokens_per_window\":{},\"maximum_unique_tokens_per_window\":{}}},",
            "\"counterfactual_context\":{{\"prefix_swap_input_differences\":{},",
            "\"suffix_swap_input_differences\":{},\"prefix_swap_argmax_changes\":{},",
            "\"suffix_swap_argmax_changes\":{},\"prefix_swap_logit_l1_q8\":{},",
            "\"suffix_swap_logit_l1_q8\":{},\"prefix_to_suffix_logit_l1_per_mille\":{},",
            "\"prefix_target_logit_abs_delta_q8\":{},\"suffix_target_logit_abs_delta_q8\":{},",
            "\"prefix_to_suffix_target_delta_per_mille\":{}}},",
            "\"residual_saturation_count\":{},\"windows\":["
        ),
        SCHEMA,
        model.model_hash(),
        model.tokenizer_hash,
        token_stream_hash,
        fnv64(&model_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&token_bytes),
        fnv64(AUDIT_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        audits.len(),
        context_tokens,
        config.rollout_tokens,
        positions,
        transitions,
        split,
        teacher_matches,
        per_mille(teacher_matches, positions),
        teacher_rank_sum / positions,
        teacher_maximum_rank,
        teacher_probability_sum_q15 / positions as i64,
        audits
            .iter()
            .map(|row| row.first_target_rank)
            .sum::<usize>()
            / audits.len(),
        audits
            .iter()
            .map(|row| row.first_target_rank)
            .max()
            .unwrap_or(0),
        free_matches,
        per_mille(free_matches, positions),
        exact_prefix_tokens,
        nonzero_prefix_windows,
        self_loop_transitions,
        per_mille(self_loop_transitions, transitions),
        audits
            .iter()
            .map(|row| row.free_running_unique_tokens)
            .min()
            .unwrap_or(0),
        audits
            .iter()
            .map(|row| row.free_running_unique_tokens)
            .max()
            .unwrap_or(0),
        prefix_input_differences,
        suffix_input_differences,
        audits
            .iter()
            .filter(|row| row.prefix_swap.argmax_changed)
            .count(),
        audits
            .iter()
            .filter(|row| row.suffix_swap.argmax_changed)
            .count(),
        prefix_l1,
        suffix_l1,
        ratio_per_mille(prefix_l1, suffix_l1),
        prefix_target_abs_delta,
        suffix_target_abs_delta,
        ratio_per_mille(prefix_target_abs_delta, suffix_target_abs_delta),
        audits
            .iter()
            .map(|row| row.residual_saturation_count)
            .sum::<usize>(),
    );
    for (index, row) in audits.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            concat!(
                "{{\"population_rank\":{},\"first_target\":{},\"native_top\":{},",
                "\"prefix_swapped_top\":{},\"suffix_swapped_top\":{},",
                "\"first_target_rank\":{},\"first_target_probability_q15\":{},",
                "\"teacher_forced_matches\":{},\"teacher_forced_mean_rank\":{},",
                "\"teacher_forced_maximum_rank\":{},",
                "\"teacher_forced_mean_target_probability_q15\":{},",
                "\"free_running_matches\":{},",
                "\"free_running_exact_prefix_tokens\":{},\"free_running_unique_tokens\":{},",
                "\"free_running_self_loop_transitions\":{},",
                "\"prefix_swap\":{{\"input_differences\":{},\"differing_logits\":{},",
                "\"logit_l1_q8\":{},",
                "\"target_logit_delta_q8\":{},\"target_probability_delta_q15\":{},",
                "\"argmax_changed\":{}}},",
                "\"suffix_swap\":{{\"input_differences\":{},\"differing_logits\":{},",
                "\"logit_l1_q8\":{},",
                "\"target_logit_delta_q8\":{},\"target_probability_delta_q15\":{},",
                "\"argmax_changed\":{}}},",
                "\"generated_token_hash\":\"0x{:016x}\",\"residual_saturation_count\":{},",
                "\"reference_tokens\":"
            ),
            row.population_rank,
            row.first_target,
            row.native_top,
            row.prefix_swapped_top,
            row.suffix_swapped_top,
            row.first_target_rank,
            row.first_target_probability_q15,
            row.teacher_forced_matches,
            row.teacher_forced_rank_sum / config.rollout_tokens,
            row.teacher_forced_maximum_rank,
            row.teacher_forced_target_probability_sum_q15 / config.rollout_tokens as i64,
            row.free_running_matches,
            row.free_running_exact_prefix_tokens,
            row.free_running_unique_tokens,
            row.free_running_self_loop_transitions,
            row.prefix_swap_input_differences,
            row.prefix_swap.differing_logits,
            row.prefix_swap.logit_l1_q8,
            row.prefix_swap.target_logit_delta_q8,
            row.prefix_swap.target_probability_delta_q15,
            row.prefix_swap.argmax_changed,
            row.suffix_swap_input_differences,
            row.suffix_swap.differing_logits,
            row.suffix_swap.logit_l1_q8,
            row.suffix_swap.target_logit_delta_q8,
            row.suffix_swap.target_probability_delta_q15,
            row.suffix_swap.argmax_changed,
            row.generated_token_hash,
            row.residual_saturation_count,
        )?;
        write_u32_array(&mut output, &row.reference_tokens)?;
        output.push_str(",\"generated_tokens\":");
        write_u32_array(&mut output, &row.generated_tokens)?;
        output.push('}');
    }
    output.push_str("]}\n");
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
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

fn select_token(logits: &[i32]) -> usize {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, _)| !is_masked(*token))
        .max_by(|(left_token, left), (right_token, right)| {
            left.cmp(right).then_with(|| right_token.cmp(left_token))
        })
        .map(|(token, _)| token)
        .unwrap_or(0)
}

fn token_rank(logits: &[i32], target: usize) -> usize {
    if target >= logits.len() || is_masked(target) {
        return 0;
    }
    let target_logit = logits[target];
    1 + logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, logit)| {
            !is_masked(*token)
                && (*logit > target_logit || (*logit == target_logit && *token < target))
        })
        .count()
}

fn compare_logits(
    native: &ProductionForward,
    counterfactual: &ProductionForward,
    target: usize,
) -> LogitComparison {
    let mut differing_logits = 0_usize;
    let mut logit_l1_q8 = 0_u64;
    for (&left, &right) in native.logits_q8.iter().zip(&counterfactual.logits_q8) {
        differing_logits = differing_logits.saturating_add(usize::from(left != right));
        logit_l1_q8 =
            logit_l1_q8.saturating_add((i64::from(left) - i64::from(right)).unsigned_abs());
    }
    LogitComparison {
        differing_logits,
        logit_l1_q8,
        target_logit_delta_q8: i64::from(native.logits_q8[target])
            - i64::from(counterfactual.logits_q8[target]),
        target_probability_delta_q15: i64::from(native.probabilities_q15[target])
            - i64::from(counterfactual.probabilities_q15[target]),
        argmax_changed: select_token(&native.logits_q8) != select_token(&counterfactual.logits_q8),
    }
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
                            population_rank,
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

fn per_mille(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1000) / denominator
    }
}

fn ratio_per_mille(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        0
    } else {
        ((u128::from(numerator) * 1000) / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
    }
}

fn write_u32_array(output: &mut String, values: &[u32]) -> std::fmt::Result {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(output, "{value}")?;
    }
    output.push(']');
    Ok(())
}

fn hash_u32(values: &[u32]) -> u64 {
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
    let mut tokenizer = None;
    let mut tokens = None;
    let mut model = None;
    let mut trace = None;
    let mut context_tokens = None;
    let mut rollout_tokens = 16_usize;
    let mut max_windows = 12_usize;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Peekable<_>, option: &str| {
            args.next()
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--tokens" => tokens = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--model" => model = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--trace" => trace = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--context-tokens" => context_tokens = Some(value(&mut args, &arg)?.parse()?),
            "--rollout-tokens" => rollout_tokens = value(&mut args, &arg)?.parse()?,
            "--max-windows" => max_windows = value(&mut args, &arg)?.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    Ok(Config {
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        model: model.ok_or("--model is required")?,
        trace: trace.ok_or("--trace is required")?,
        context_tokens,
        rollout_tokens,
        max_windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_masks_control_tokens_and_breaks_ties_by_token() {
        let mut logits = vec![0; (BOS_TOKEN_ID.max(PAD_TOKEN_ID) as usize) + 2];
        logits[BOS_TOKEN_ID as usize] = 100;
        logits[PAD_TOKEN_ID as usize] = 100;
        logits[6] = 7;
        logits[7] = 7;
        assert_eq!(select_token(&logits), 6);
        assert_eq!(token_rank(&logits, 7), 2);
    }

    #[test]
    fn continuation_windows_are_spread_to_population_endpoints() {
        let tokens = vec![BOS_TOKEN_ID, 4, 5, 6, 7, 8, 9, 10, EOS_TOKEN_ID];
        let windows = spread_continuation_windows(&tokens, 2, 2, 2);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].population_rank, 0);
        assert_eq!(windows[0].context, vec![4, 5]);
        assert_eq!(windows[0].continuation, vec![6, 7]);
        assert_eq!(windows[1].population_rank, 3);
        assert_eq!(windows[1].context, vec![7, 8]);
        assert_eq!(windows[1].continuation, vec![9, 10]);
    }

    #[test]
    fn ratios_are_integer_and_zero_safe() {
        assert_eq!(per_mille(1, 4), 250);
        assert_eq!(per_mille(1, 0), 0);
        assert_eq!(ratio_per_mille(1, 4), 250);
        assert_eq!(ratio_per_mille(1, 0), 0);
    }
}
