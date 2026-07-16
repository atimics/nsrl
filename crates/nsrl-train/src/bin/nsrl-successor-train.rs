#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nsrl_core::{
    DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, base2_softmax_nll_millibits, round_shift_rhu_i64,
};
use nsrl_train::{
    MINI_TRANSFORMER_D_MODEL, MiniTransformerAttentionKind, MiniTransformerMlpModel,
    MiniTransformerPositionPolicy,
    mini_transformer_next_token_row_with_attention_kind_position_policy,
};

const BYTE_CLASSES: usize = 256;
const DEFAULT_CONTEXT: usize = 64;
const DEFAULT_MAX_PASSES: usize = 8;
const TRANSFORMER_LAYERS: usize = 2;
const CONSTANT_FEATURE_Q15: i16 = i16::MAX;
// Output logits are approximately weight * 128 in Q8. A 29-step span stays
// strictly inside the 15-bit support of base2_exp_neg_q15.
const MIN_OUTPUT_WEIGHT: i8 = -29;
const MAX_OUTPUT_WEIGHT: i8 = 0;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    train: PathBuf,
    out: PathBuf,
    trace: PathBuf,
    context: usize,
    max_passes: usize,
}

#[derive(Debug, Clone, Copy)]
struct TrainingSummary {
    initial_nll_millibits: u64,
    final_nll_millibits: u64,
    passes: usize,
    accepted_coordinate_updates: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-successor-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let train = fs::read(&config.train)?;
    if train.len() <= config.context {
        return Err("training corpus must contain at least one canonical target".into());
    }
    let counts = target_counts(&train, config.context);
    let targets = train.len() - config.context;
    let (weights, summary) = train_canonical_head(&counts, config.max_passes)?;
    let model = materialize_model(config.context, &weights)?;
    let context = &train[..config.context];
    let row = mini_transformer_next_token_row_with_attention_kind_position_policy(
        &model,
        context,
        MiniTransformerAttentionKind::Linear,
        MiniTransformerPositionPolicy::Nope,
    )?;
    let expected_logits = logits_for_weights(&weights);
    if row.logits_q8 != expected_logits {
        return Err("materialized native model logits differ from the optimized head".into());
    }
    let zero_probability_classes = zero_probability_classes(&row.logits_q8)?;
    if zero_probability_classes != 0 {
        return Err("trained head left byte classes outside canonical exponent support".into());
    }
    let final_replay_nll = canonical_nll(&counts, &row.logits_q8)?;
    if final_replay_nll != summary.final_nll_millibits {
        return Err("materialized native model changed the canonical training objective".into());
    }
    let artifact = model.try_to_bytes()?;
    write_parent(&config.out)?;
    write_parent(&config.trace)?;
    fs::write(&config.out, &artifact)?;
    let trace = format!(
        concat!(
            "{{\"schema\":\"nsrl.integer_transformer_successor_train.v1\",",
            "\"objective\":{{\"id\":\"integer_base2_softmax_nll_millibits\",",
            "\"partition\":\"train\",\"context\":{},\"targets\":{},",
            "\"zero_probability_floor_millibits\":{}}},",
            "\"method\":{{\"name\":\"deterministic_constrained_coordinate_descent\",",
            "\"transformer_layers\":{},",
            "\"constant_residual_feature_q15\":{},\"minimum_output_weight\":{},",
            "\"maximum_output_weight\":{},\"maximum_passes\":{},\"passes\":{},",
            "\"accepted_coordinate_updates\":{}}},",
            "\"metrics\":{{\"uniform_nll_millibits\":{},",
            "\"initial_nll_millibits\":{},\"final_nll_millibits\":{},",
            "\"zero_probability_classes\":{}}},",
            "\"assistance\":{{\"suffix_memory\":false,\"retrieval\":false,",
            "\"routing_oracle\":false,\"heldout_targets_read\":false}},",
            "\"bindings\":{{\"train_fnv64\":\"0x{:016x}\",",
            "\"model_hash\":\"0x{:016x}\",\"artifact_fnv64\":\"0x{:016x}\"}}}}\n"
        ),
        config.context,
        targets,
        DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
        TRANSFORMER_LAYERS,
        CONSTANT_FEATURE_Q15,
        MIN_OUTPUT_WEIGHT,
        MAX_OUTPUT_WEIGHT,
        config.max_passes,
        summary.passes,
        summary.accepted_coordinate_updates,
        u64::try_from(targets)?.saturating_mul(8_000),
        summary.initial_nll_millibits,
        summary.final_nll_millibits,
        zero_probability_classes,
        fnv64(&train),
        model.model_hash(),
        fnv64(&artifact),
    );
    fs::write(&config.trace, &trace)?;
    print!("{trace}");
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut train = None;
    let mut out = None;
    let mut trace = None;
    let mut context = DEFAULT_CONTEXT;
    let mut max_passes = DEFAULT_MAX_PASSES;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train = Some(PathBuf::from(required(&mut args, "--train")?)),
            "--out" => out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--context" => context = required(&mut args, "--context")?.parse()?,
            "--max-passes" => max_passes = required(&mut args, "--max-passes")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-successor-train --train PATH --out PATH --trace PATH [--context 64] [--max-passes 8]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if context == 0 || max_passes == 0 {
        return Err("context and max-passes must be positive".into());
    }
    Ok(Config {
        train: train.ok_or("--train is required")?,
        out: out.ok_or("--out is required")?,
        trace: trace.ok_or("--trace is required")?,
        context,
        max_passes,
    })
}

fn required(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value").into())
}

fn target_counts(tokens: &[u8], context: usize) -> [u64; BYTE_CLASSES] {
    let mut counts = [0_u64; BYTE_CLASSES];
    for &target in &tokens[context..] {
        counts[usize::from(target)] = counts[usize::from(target)].saturating_add(1);
    }
    counts
}

fn initial_weights(counts: &[u64; BYTE_CLASSES]) -> [i8; BYTE_CLASSES] {
    let maximum_log2 = counts.iter().copied().max().unwrap_or(1).max(1).ilog2();
    let mut weights = [MIN_OUTPUT_WEIGHT; BYTE_CLASSES];
    for (weight, &count) in weights.iter_mut().zip(counts.iter()) {
        if count == 0 {
            continue;
        }
        let log2_gap = maximum_log2.saturating_sub(count.ilog2());
        let candidate = -i16::try_from(log2_gap.saturating_mul(2)).unwrap_or(i16::MAX);
        *weight = i8::try_from(
            candidate.clamp(i16::from(MIN_OUTPUT_WEIGHT), i16::from(MAX_OUTPUT_WEIGHT)),
        )
        .expect("clamped output weight");
    }
    weights
}

fn train_canonical_head(
    counts: &[u64; BYTE_CLASSES],
    max_passes: usize,
) -> Result<([i8; BYTE_CLASSES], TrainingSummary), Box<dyn std::error::Error>> {
    let mut weights = initial_weights(counts);
    let mut best_nll = canonical_nll(counts, &logits_for_weights(&weights))?;
    let initial_nll = best_nll;
    let mut accepted_coordinate_updates = 0_usize;
    let mut passes = 0_usize;
    for _ in 0..max_passes {
        passes += 1;
        let mut changed = false;
        for index in 0..BYTE_CLASSES {
            let original = weights[index];
            let mut selected = original;
            let mut selected_nll = best_nll;
            for delta in [-1_i8, 1_i8] {
                let candidate = original.saturating_add(delta);
                if !(MIN_OUTPUT_WEIGHT..=MAX_OUTPUT_WEIGHT).contains(&candidate) {
                    continue;
                }
                weights[index] = candidate;
                let candidate_logits = logits_for_weights(&weights);
                if zero_probability_classes(&candidate_logits)? != 0 {
                    continue;
                }
                let candidate_nll = canonical_nll(counts, &candidate_logits)?;
                if candidate_nll < selected_nll {
                    selected = candidate;
                    selected_nll = candidate_nll;
                }
            }
            weights[index] = selected;
            if selected != original {
                best_nll = selected_nll;
                accepted_coordinate_updates += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok((
        weights,
        TrainingSummary {
            initial_nll_millibits: initial_nll,
            final_nll_millibits: best_nll,
            passes,
            accepted_coordinate_updates,
        },
    ))
}

fn logits_for_weights(weights: &[i8; BYTE_CLASSES]) -> [i32; BYTE_CLASSES] {
    let mut logits = [0_i32; BYTE_CLASSES];
    for (logit, &weight) in logits.iter_mut().zip(weights.iter()) {
        let product = i64::from(CONSTANT_FEATURE_Q15) * i64::from(weight);
        let scaled =
            round_shift_rhu_i64(product, 8).clamp(i64::from(i16::MIN), i64::from(i16::MAX));
        *logit = scaled as i32;
    }
    logits
}

fn canonical_nll(
    counts: &[u64; BYTE_CLASSES],
    logits: &[i32; BYTE_CLASSES],
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total = 0_u64;
    for (target, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let loss =
            base2_softmax_nll_millibits(logits, target, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS)
                .ok_or("canonical NLL rejected successor logits")?;
        total = total
            .checked_add(
                count
                    .checked_mul(loss)
                    .ok_or("canonical NLL count overflow")?,
            )
            .ok_or("canonical NLL total overflow")?;
    }
    Ok(total)
}

fn zero_probability_classes(
    logits: &[i32; BYTE_CLASSES],
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut zero = 0_usize;
    for target in 0..BYTE_CLASSES {
        let loss =
            base2_softmax_nll_millibits(logits, target, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS)
                .ok_or("canonical NLL rejected successor logits")?;
        zero += usize::from(loss == DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS);
    }
    Ok(zero)
}

fn materialize_model(
    context: usize,
    weights: &[i8; BYTE_CLASSES],
) -> Result<MiniTransformerMlpModel, Box<dyn std::error::Error>> {
    let mut model =
        MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(context, TRANSFORMER_LAYERS)?;
    model.embeddings.fill(0);
    for row in model.embeddings.chunks_exact_mut(MINI_TRANSFORMER_D_MODEL) {
        row[0] = CONSTANT_FEATURE_Q15;
    }
    model.position_embeddings.fill(0);
    model.attention_rms_weights.clear();
    model.mlp_rms_weights.clear();
    model.q_weights.fill(0);
    model.k_weights.fill(0);
    model.v_weights.fill(0);
    model.o_weights.fill(0);
    model.up_weights.fill(0);
    model.gate_weights.fill(0);
    model.down_weights.fill(0);
    model.output_weights.fill(0);
    for (token, &weight) in weights.iter().enumerate() {
        model.output_weights[token * MINI_TRANSFORMER_D_MODEL] = weight;
    }
    Ok(MiniTransformerMlpModel::from_bytes(&model.try_to_bytes()?)?)
}

fn write_parent(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_head_beats_uniform_without_zero_classes() {
        let mut counts = [0_u64; BYTE_CLASSES];
        counts[usize::from(b' ')] = 500;
        counts[usize::from(b'e')] = 250;
        counts[usize::from(b't')] = 125;
        counts[usize::from(b'\n')] = 62;
        let targets = counts.iter().sum::<u64>();
        let (weights, summary) = train_canonical_head(&counts, 8).expect("trained head");
        let logits = logits_for_weights(&weights);
        assert_eq!(zero_probability_classes(&logits).expect("zero count"), 0);
        assert!(summary.final_nll_millibits < targets * 8_000);
    }

    #[test]
    fn materialized_model_replays_optimized_logits() {
        let mut weights = [MIN_OUTPUT_WEIGHT; BYTE_CLASSES];
        weights[usize::from(b' ')] = 0;
        weights[usize::from(b'e')] = -2;
        let model = materialize_model(DEFAULT_CONTEXT, &weights).expect("model");
        let row = mini_transformer_next_token_row_with_attention_kind_position_policy(
            &model,
            &[b'x'; DEFAULT_CONTEXT],
            MiniTransformerAttentionKind::Linear,
            MiniTransformerPositionPolicy::Nope,
        )
        .expect("forward");
        assert_eq!(row.logits_q8, logits_for_weights(&weights));
        assert!(model.position_embeddings.iter().all(|&value| value == 0));
    }
}
