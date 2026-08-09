#![deny(unsafe_code)]

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_core::{
    DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, base2_exp_neg_q15, base2_softmax_nll_millibits,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, SubwordTokenizer};
use nsrl_train::production::{
    ProductionModelV1, decode_bound_token_stream,
    evaluate_production_model_canonical_nll_default_floor, forward_production_model,
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
    guard_update_windows: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct CompositionEvaluation {
    model_hash: u64,
    windows: usize,
    total_nll_millibits: u64,
    mistakes: usize,
    zero_probability_windows: usize,
    residual_saturation_count: usize,
}

#[derive(Debug)]
struct GuardSurface {
    windows: Vec<(Vec<u32>, u32)>,
    rank_hash: u64,
    update_windows: usize,
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

    let guard_surface = config
        .guard_update_windows
        .map(|update_windows| {
            guard_surface(
                &tokens,
                config.context_tokens,
                update_windows,
                config.max_windows,
            )
        })
        .transpose()?;
    let source_eval = evaluate_composition(
        &source,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
        guard_surface.as_ref(),
    )?;
    let subsets = if guard_surface.is_some() {
        (0_u8..16)
            .map(|mask| {
                let groups = GROUPS
                    .iter()
                    .enumerate()
                    .filter(|&(index, _)| mask & (1 << index) != 0)
                    .map(|(_, group)| *group)
                    .collect::<Vec<_>>();
                let id = if mask == 0 {
                    "source".to_string()
                } else if mask == 15 {
                    "candidate".to_string()
                } else {
                    groups.join("_plus_")
                };
                (id, groups)
            })
            .collect::<Vec<_>>()
    } else {
        vec![
            ("source".to_string(), Vec::new()),
            ("embeddings_only".to_string(), vec!["embeddings"]),
            ("k_only".to_string(), vec!["k"]),
            ("v_only".to_string(), vec!["v"]),
            ("o_only".to_string(), vec!["o"]),
            ("without_embeddings".to_string(), vec!["k", "v", "o"]),
            ("without_k".to_string(), vec!["embeddings", "v", "o"]),
            ("without_v".to_string(), vec!["embeddings", "k", "o"]),
            ("without_o".to_string(), vec!["embeddings", "k", "v"]),
            ("candidate".to_string(), GROUPS.to_vec()),
        ]
    };
    let mut rows = String::new();
    for (index, (id, groups)) in subsets.iter().enumerate() {
        let model = if groups.is_empty() {
            source.clone()
        } else if groups.len() == GROUPS.len() {
            candidate.clone()
        } else {
            compose(&source, &candidate, groups)?
        };
        let evaluation = evaluate_composition(
            &model,
            &tokens,
            token_stream_hash,
            config.context_tokens,
            config.max_windows,
            guard_surface.as_ref(),
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

    let evaluation_surface = guard_surface.as_ref().map_or_else(
        || {
            format!(
                "\"partition\":\"development\",\"metric\":\"canonical_integer_base2_softmax_nll_millibits\",\"context_tokens\":{},\"windows\":{}",
                config.context_tokens, source_eval.windows,
            )
        },
        |guard| {
            format!(
                "\"partition\":\"training_descent_guard\",\"metric\":\"canonical_integer_base2_softmax_nll_millibits\",\"context_tokens\":{},\"windows\":{},\"window_selection\":\"fixed_global_ranks_disjoint_from_spread_update_windows\",\"update_windows\":{},\"window_rank_hash\":\"0x{:016x}\",\"update_window_overlap_count\":0",
                config.context_tokens,
                source_eval.windows,
                guard.update_windows,
                guard.rank_hash,
            )
        },
    );

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
            "\"evaluation\":{{{}}},",
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
        evaluation_surface,
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
    let mut guard_update_windows = None;
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
            "--guard-update-windows" => {
                guard_update_windows = Some(args.next().ok_or_else(value)?.parse()?)
            }
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
        guard_update_windows,
    })
}

fn evaluate_composition(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
    guard_surface: Option<&GuardSurface>,
) -> Result<CompositionEvaluation, Box<dyn std::error::Error>> {
    if let Some(guard) = guard_surface {
        let mut total_nll_millibits = 0_u64;
        let mut mistakes = 0_usize;
        let mut zero_probability_windows = 0_usize;
        let mut residual_saturation_count = 0_usize;
        for (context, target) in &guard.windows {
            let output = forward_production_model(model, context)?;
            let target = *target as usize;
            let predicted = output
                .logits_q8
                .iter()
                .enumerate()
                .max_by_key(|&(index, &value)| (value, Reverse(index)))
                .map(|(index, _)| index)
                .unwrap_or(0);
            mistakes = mistakes.saturating_add(usize::from(predicted != target));
            let maximum = output
                .logits_q8
                .iter()
                .copied()
                .max()
                .ok_or("empty logits")?;
            zero_probability_windows = zero_probability_windows.saturating_add(usize::from(
                base2_exp_neg_q15(output.logits_q8[target].saturating_sub(maximum)) == 0,
            ));
            let loss = base2_softmax_nll_millibits(
                &output.logits_q8,
                target,
                DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
            )
            .ok_or("canonical guard NLL failed")?;
            total_nll_millibits = total_nll_millibits
                .checked_add(loss)
                .ok_or("canonical guard NLL overflow")?;
            residual_saturation_count = residual_saturation_count
                .checked_add(output.residual_saturation_count)
                .ok_or("guard residual saturation overflow")?;
        }
        return Ok(CompositionEvaluation {
            model_hash: model.model_hash(),
            windows: guard.windows.len(),
            total_nll_millibits,
            mistakes,
            zero_probability_windows,
            residual_saturation_count,
        });
    }
    let evaluation = evaluate_production_model_canonical_nll_default_floor(
        model,
        tokens,
        token_stream_hash,
        context_tokens,
        max_windows,
    )?;
    Ok(CompositionEvaluation {
        model_hash: evaluation.model_hash,
        windows: evaluation.windows,
        total_nll_millibits: evaluation.total_nll_millibits,
        mistakes: evaluation.mistakes,
        zero_probability_windows: evaluation.zero_probability_windows,
        residual_saturation_count: evaluation.residual_saturation_count,
    })
}

fn guard_surface(
    tokens: &[u32],
    context_tokens: usize,
    update_windows: usize,
    guard_windows: usize,
) -> Result<GuardSurface, Box<dyn std::error::Error>> {
    let total_windows = document_window_count(tokens, context_tokens);
    let selected_updates = update_windows.min(total_windows);
    if selected_updates == 0 || guard_windows == 0 || selected_updates == total_windows {
        return Err("invalid guard surface geometry".into());
    }
    let update_ranks = if selected_updates == 1 {
        vec![total_windows / 2]
    } else {
        (0..selected_updates)
            .map(|index| {
                ((index as u128) * ((total_windows - 1) as u128) / ((selected_updates - 1) as u128))
                    as usize
            })
            .collect::<Vec<_>>()
    };
    let excluded = update_ranks.iter().copied().collect::<BTreeSet<_>>();
    if total_windows.saturating_sub(excluded.len()) < guard_windows {
        return Err("not enough disjoint guard windows".into());
    }
    let mut ranks = BTreeSet::new();
    for index in 0..guard_windows {
        let target = (((index as u128 * 2 + 1) * total_windows as u128)
            / (guard_windows as u128 * 2)) as usize;
        let target = target.min(total_windows - 1);
        let rank = (0..total_windows)
            .find_map(|distance| {
                if let Some(rank) = target.checked_sub(distance)
                    && !excluded.contains(&rank)
                    && !ranks.contains(&rank)
                {
                    return Some(rank);
                }
                if distance > 0 {
                    let rank = target.saturating_add(distance);
                    if rank < total_windows && !excluded.contains(&rank) && !ranks.contains(&rank) {
                        return Some(rank);
                    }
                }
                None
            })
            .ok_or("guard rank selection failed")?;
        ranks.insert(rank);
    }
    let ranks = ranks.into_iter().collect::<Vec<_>>();
    let windows = document_windows_at_ranks(tokens, context_tokens, &ranks);
    if windows.len() != guard_windows {
        return Err("guard surface materialization failed".into());
    }
    let mut rank_bytes = Vec::with_capacity(ranks.len() * 8);
    for rank in ranks {
        rank_bytes.extend_from_slice(&(rank as u64).to_le_bytes());
    }
    Ok(GuardSurface {
        windows,
        rank_hash: fnv64(&rank_bytes),
        update_windows: selected_updates,
    })
}

fn document_window_count(tokens: &[u32], context_tokens: usize) -> usize {
    let mut total = 0_usize;
    let mut document_tokens = 0_usize;
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document_tokens = 0;
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active {
                total = total.saturating_add(document_tokens.saturating_sub(context_tokens));
            }
            document_tokens = 0;
            active = false;
        } else if active {
            document_tokens = document_tokens.saturating_add(1);
        }
    }
    total
}

fn document_windows_at_ranks(
    tokens: &[u32],
    context_tokens: usize,
    ranks: &[usize],
) -> Vec<(Vec<u32>, u32)> {
    let mut windows = Vec::with_capacity(ranks.len());
    let mut rank_cursor = 0_usize;
    let mut current_rank = 0_usize;
    let mut document = Vec::new();
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active && document.len() > context_tokens {
                for start in 0..document.len() - context_tokens {
                    if rank_cursor < ranks.len() && current_rank == ranks[rank_cursor] {
                        windows.push((
                            document[start..start + context_tokens].to_vec(),
                            document[start + context_tokens],
                        ));
                        rank_cursor += 1;
                        if rank_cursor == ranks.len() {
                            return windows;
                        }
                    }
                    current_rank = current_rank.saturating_add(1);
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
    use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

    use super::{guard_surface, signed_delta};

    #[test]
    fn signed_delta_preserves_direction() {
        assert_eq!(signed_delta(12, 10), 2);
        assert_eq!(signed_delta(8, 10), -2);
    }

    #[test]
    fn guard_surface_is_deterministic_and_excludes_spread_update_ranks() {
        let tokens = [BOS_TOKEN_ID, 10, 11, 12, 13, 14, 15, 16, 17, EOS_TOKEN_ID];
        let left = guard_surface(&tokens, 2, 3, 3).expect("guard surface");
        let right = guard_surface(&tokens, 2, 3, 3).expect("guard replay");

        assert_eq!(left.windows, right.windows);
        assert_eq!(left.rank_hash, right.rank_hash);
        assert_eq!(left.windows.len(), 3);
        assert_eq!(left.update_windows, 3);
    }
}
