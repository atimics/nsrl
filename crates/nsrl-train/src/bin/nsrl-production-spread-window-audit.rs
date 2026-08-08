#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID, SubwordTokenizer};
use nsrl_train::production::decode_bound_token_stream;

const SCHEMA: &str = "nsrl.production_spread_window_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-spread-window-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    tokenizer: PathBuf,
    tokens: PathBuf,
    trace: PathBuf,
    context_tokens: usize,
    max_windows: usize,
    start_window: usize,
    windows: usize,
    targets_per_window: usize,
    batch_windows: usize,
}

#[derive(Debug)]
struct SelectedWindow {
    selected_index: usize,
    population_rank: usize,
    document_index: usize,
    document_start: usize,
    document_tokens: usize,
    context: Vec<u32>,
    target: u32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-spread-window-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.context_tokens == 0
        || config.max_windows == 0
        || config.windows == 0
        || config.targets_per_window == 0
        || config.targets_per_window > config.context_tokens
        || config.batch_windows == 0
        || config.start_window.saturating_add(config.windows) > config.max_windows
    {
        return Err("invalid spread-window audit geometry".into());
    }
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let token_bytes = fs::read(&config.tokens)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let tokenizer_hash = tokenizer.tokenizer_hash();
    let (tokens, token_stream_hash) =
        decode_bound_token_stream(&token_bytes, tokenizer_hash, tokenizer.vocab_size())?;
    let (population_windows, selected) = select_spread_windows(
        &tokens,
        config.context_tokens,
        config.max_windows,
        config.start_window,
        config.windows,
    );
    if selected.len() != config.windows {
        return Err("requested spread-window range is not present".into());
    }

    let mut output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"token_stream_hash\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"token_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"selection\":{{\"policy\":\"deterministic_uniform_target_rank_over_all_documents\",",
            "\"population_windows\":{},\"max_windows\":{},\"context_tokens\":{},",
            "\"start_window\":{},\"windows\":{},\"targets_per_window\":{},",
            "\"batch_windows\":{}}},\"selected\":["
        ),
        SCHEMA,
        tokenizer_hash,
        token_stream_hash,
        fnv64(&tokenizer_bytes),
        fnv64(&token_bytes),
        fnv64(AUDIT_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        population_windows,
        config.max_windows.min(population_windows),
        config.context_tokens,
        config.start_window,
        config.windows,
        config.targets_per_window,
        config.batch_windows,
    );
    for (index, window) in selected.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let suffix_start = window.context.len() + 1 - config.targets_per_window;
        let mut supervised = window.context[suffix_start..].to_vec();
        supervised.push(window.target);
        let mut decoded_tokens = window.context.clone();
        decoded_tokens.push(window.target);
        let decoded = tokenizer.decode(&decoded_tokens)?;
        let unique_context_tokens = {
            let mut unique = window.context.clone();
            unique.sort_unstable();
            unique.dedup();
            unique.len()
        };
        write!(
            output,
            concat!(
                "{{\"selected_index\":{},\"population_rank\":{},",
                "\"optimizer_step\":{},\"batch_offset\":{},",
                "\"document_index\":{},\"document_start\":{},",
                "\"document_tokens\":{},\"target\":{},",
                "\"unique_context_tokens\":{},",
                "\"context_token_fnv64\":\"0x{:016x}\",",
                "\"decoded_bytes\":{},\"decoded_fnv64\":\"0x{:016x}\",",
                "\"supervised_targets\":["
            ),
            window.selected_index,
            window.population_rank,
            window.selected_index / config.batch_windows + 1,
            window.selected_index % config.batch_windows,
            window.document_index,
            window.document_start,
            window.document_tokens,
            window.target,
            unique_context_tokens,
            fnv64_tokens(&window.context),
            decoded.len(),
            fnv64(&decoded),
        )?;
        push_u32_array(&mut output, &supervised)?;
        output.push_str("],\"context_tokens\":[");
        push_u32_array(&mut output, &window.context)?;
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

fn select_spread_windows(
    tokens: &[u32],
    context_tokens: usize,
    max_windows: usize,
    start_window: usize,
    requested_windows: usize,
) -> (usize, Vec<SelectedWindow>) {
    let mut population_windows = 0_usize;
    let mut document_tokens = 0_usize;
    let mut in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document_tokens = 0;
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document {
                population_windows = population_windows
                    .saturating_add(document_tokens.saturating_sub(context_tokens));
            }
            document_tokens = 0;
            in_document = false;
        } else if in_document {
            document_tokens = document_tokens.saturating_add(1);
        }
    }
    let selected_count = max_windows.min(population_windows);
    if selected_count == 0 {
        return (population_windows, Vec::new());
    }
    let ranks = if selected_count == 1 {
        vec![population_windows / 2]
    } else {
        (0..selected_count)
            .map(|index| {
                ((index as u128) * ((population_windows - 1) as u128)
                    / ((selected_count - 1) as u128)) as usize
            })
            .collect::<Vec<_>>()
    };
    let end_window = start_window.saturating_add(requested_windows);
    let mut result = Vec::with_capacity(requested_windows);
    let mut rank_cursor = 0_usize;
    let mut current_rank = 0_usize;
    let mut document = Vec::new();
    let mut document_index = 0_usize;
    in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document && document.len() > context_tokens {
                for start in 0..document.len() - context_tokens {
                    if rank_cursor < ranks.len() && current_rank == ranks[rank_cursor] {
                        if (start_window..end_window).contains(&rank_cursor) {
                            result.push(SelectedWindow {
                                selected_index: rank_cursor,
                                population_rank: current_rank,
                                document_index,
                                document_start: start,
                                document_tokens: document.len(),
                                context: document[start..start + context_tokens].to_vec(),
                                target: document[start + context_tokens],
                            });
                        }
                        rank_cursor += 1;
                        if rank_cursor == ranks.len() || rank_cursor == end_window {
                            return (population_windows, result);
                        }
                    }
                    current_rank = current_rank.saturating_add(1);
                }
            }
            document.clear();
            document_index = document_index.saturating_add(1);
            in_document = false;
        } else if in_document {
            document.push(token);
        }
    }
    (population_windows, result)
}

fn push_u32_array(output: &mut String, values: &[u32]) -> Result<(), std::fmt::Error> {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(output, "{value}")?;
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn fnv64_tokens(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |mut hash, token| {
        for byte in token.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut tokenizer = None;
    let mut tokens = None;
    let mut trace = None;
    let mut context_tokens = 64;
    let mut max_windows = 2048;
    let mut start_window = None;
    let mut windows = None;
    let mut targets_per_window = 8;
    let mut batch_windows = 4;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = || "missing audit argument value";
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--tokens" => tokens = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--trace" => trace = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--context-tokens" => context_tokens = args.next().ok_or_else(value)?.parse()?,
            "--max-windows" => max_windows = args.next().ok_or_else(value)?.parse()?,
            "--start-window" => start_window = Some(args.next().ok_or_else(value)?.parse()?),
            "--windows" => windows = Some(args.next().ok_or_else(value)?.parse()?),
            "--targets-per-window" => {
                targets_per_window = args.next().ok_or_else(value)?.parse()?
            }
            "--batch-windows" => batch_windows = args.next().ok_or_else(value)?.parse()?,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(Config {
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        trace: trace.ok_or("--trace is required")?,
        context_tokens,
        max_windows,
        start_window: start_window.ok_or("--start-window is required")?,
        windows: windows.ok_or("--windows is required")?,
        targets_per_window,
        batch_windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_range_matches_production_spread_ranks() {
        let tokens = [
            BOS_TOKEN_ID,
            10,
            11,
            12,
            13,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            20,
            21,
            22,
            23,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            30,
            31,
            32,
            33,
            EOS_TOKEN_ID,
        ];
        let (population, selected) = select_spread_windows(&tokens, 2, 3, 1, 2);
        assert_eq!(population, 6);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].selected_index, 1);
        assert_eq!(selected[0].population_rank, 2);
        assert_eq!(selected[0].document_index, 1);
        assert_eq!(selected[0].context, [20, 21]);
        assert_eq!(selected[0].target, 22);
        assert_eq!(selected[1].selected_index, 2);
        assert_eq!(selected[1].population_rank, 5);
        assert_eq!(selected[1].document_index, 2);
        assert_eq!(selected[1].context, [31, 32]);
        assert_eq!(selected[1].target, 33);
    }
}
