#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_core::{round_shift_rhu_i64, saturate_i16};
use nsrl_train::{
    MINI_TRANSFORMER_D_MODEL, MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig,
    MiniTransformerMlpModel, MiniTransformerMlpWindowEvalRecord, MiniTransformerPositionPolicy,
    evaluate_mini_transformer_mlp_windows, mini_transformer_output_from_hidden_q15,
    mini_transformer_output_gradient_to_hidden_q15,
};

const MAGIC: &[u8; 8] = b"NSRLLE1\n";
const VOCAB: usize = 256;
const DEFAULT_PROJECTION_SEED: u64 = 0x4c4f_5752_414e_4b31;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LowRankExpert {
    trunk_model_hash: u64,
    rank: usize,
    projection_seed: u64,
    expansion_weights_q15: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Metrics {
    windows: usize,
    mistakes: usize,
    probability_error_q15: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrainStats {
    optimizer_steps: usize,
    weight_delta_l1: u64,
    weight_saturation_count: usize,
    hidden_saturation_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-low-rank-expert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => run_train(args),
        Some("eval") => run_eval(args),
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        _ => Err("expected train or eval".into()),
    }
}

fn run_train(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_out = None;
    let mut trace_out = None;
    let mut epochs = 1_usize;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut batch_windows = usize::MAX;
    let mut learning_rate = 1024_i64;
    let mut learning_rate_shift = 0_u8;
    let mut rank = 4_usize;
    let mut projection_seed = DEFAULT_PROJECTION_SEED;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => expert_out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--epochs" => epochs = required(&mut args, "--epochs")?.parse()?,
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--batch-windows" => batch_windows = required(&mut args, "--batch-windows")?.parse()?,
            "--learning-rate" => learning_rate = required(&mut args, "--learning-rate")?.parse()?,
            "--learning-rate-shift" => {
                learning_rate_shift = required(&mut args, "--learning-rate-shift")?.parse()?
            }
            "--rank" => rank = required(&mut args, "--rank")?.parse()?,
            "--projection-seed" => {
                projection_seed = required(&mut args, "--projection-seed")?.parse()?
            }
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    if epochs == 0
        || stride == 0
        || batch_windows == 0
        || learning_rate_shift > 30
        || !(1..=16_384).contains(&learning_rate)
        || rank == 0
        || rank > 32
    {
        return Err("invalid training configuration".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let mut expert = LowRankExpert::new(model.model_hash(), rank, projection_seed)?;
    let initial = evaluate_expert(&records, &tokens, &model, &expert)?;
    let stats = train_expert(
        &records,
        &tokens,
        &model,
        &mut expert,
        epochs,
        batch_windows,
        learning_rate,
        learning_rate_shift,
    )?;
    let final_metrics = evaluate_expert(&records, &tokens, &model, &expert)?;
    let artifact = expert.to_bytes()?;
    fs::write(expert_out.ok_or("--out is required")?, &artifact)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_low_rank_expert_train.v1\",\"expert_type\":\"fixed_projection_low_rank_hidden_residual_q15\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"parameter_count\":{},\"config\":{{\"rank\":{},\"projection_seed\":{},\"epochs\":{},\"stride\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate\":{},\"learning_rate_shift\":{},\"objective\":\"probability_error\"}},\"initial\":{},\"final\":{},\"updates\":{{\"optimizer_steps\":{},\"weight_delta_l1\":{},\"weight_saturation_count\":{},\"hidden_saturation_count\":{}}}}}\n",
            expert.trunk_model_hash,
            fnv64(&artifact),
            expert.expansion_weights_q15.len(),
            rank,
            projection_seed,
            epochs,
            stride,
            optional_usize_json(max_windows),
            batch_windows.min(records.len()),
            learning_rate,
            learning_rate_shift,
            metrics_json(initial),
            metrics_json(final_metrics),
            stats.optimizer_steps,
            stats.weight_delta_l1,
            stats.weight_saturation_count,
            stats.hidden_saturation_count,
        ),
    )?;
    Ok(())
}

fn run_eval(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut expert_path = None;
    let mut trace_out = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--expert" => expert_path = Some(PathBuf::from(required(&mut args, "--expert")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            other => return Err(format!("unknown eval argument: {other}").into()),
        }
    }
    if stride == 0 {
        return Err("stride must be positive".into());
    }
    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let expert_bytes = fs::read(expert_path.ok_or("--expert is required")?)?;
    let expert = LowRankExpert::from_bytes(&expert_bytes)?;
    if expert.trunk_model_hash != model.model_hash() {
        return Err("low-rank expert is bound to a different trunk".into());
    }
    let records = evaluate_records(&tokens, &model, stride, max_windows)?;
    let metrics = evaluate_expert(&records, &tokens, &model, &expert)?;
    fs::write(
        trace_out.ok_or("--trace is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_low_rank_expert_eval.v1\",\"trunk_model_hash\":\"0x{:016x}\",\"artifact_hash\":\"0x{:016x}\",\"rank\":{},\"stride\":{},\"max_windows\":{},\"metrics\":{}}}\n",
            model.model_hash(),
            fnv64(&expert_bytes),
            expert.rank,
            stride,
            optional_usize_json(max_windows),
            metrics_json(metrics),
        ),
    )?;
    Ok(())
}

fn print_help() {
    println!("Usage:");
    println!(
        "  nsrl-mini-transformer-low-rank-expert train --tokens PATH --model PATH --out PATH --trace PATH [--rank N] [--projection-seed N] [--epochs N] [--stride N] [--max-windows N] [--batch-windows N] [--learning-rate N] [--learning-rate-shift N]"
    );
    println!(
        "  nsrl-mini-transformer-low-rank-expert eval --tokens PATH --model PATH --expert PATH --trace PATH [--stride N] [--max-windows N]"
    );
}

fn evaluate_records(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    stride: usize,
    max_windows: Option<usize>,
) -> Result<Vec<MiniTransformerMlpWindowEvalRecord>, Box<dyn std::error::Error>> {
    Ok(evaluate_mini_transformer_mlp_windows(
        tokens,
        model,
        MiniTransformerMlpEvalConfig {
            seq_len: model.context_seq_len,
            stride,
            max_windows,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
        },
    )?)
}

impl LowRankExpert {
    fn new(
        trunk_model_hash: u64,
        rank: usize,
        projection_seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let count = MINI_TRANSFORMER_D_MODEL
            .checked_mul(rank)
            .ok_or("low-rank parameter overflow")?;
        Ok(Self {
            trunk_model_hash,
            rank,
            projection_seed,
            expansion_weights_q15: vec![0; count],
        })
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.rank == 0
            || self.rank > 32
            || self.expansion_weights_q15.len() != MINI_TRANSFORMER_D_MODEL * self.rank
        {
            return Err("invalid low-rank expert shape".into());
        }
        Ok(())
    }

    fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        out.extend_from_slice(&(self.rank as u32).to_le_bytes());
        out.extend_from_slice(&self.projection_seed.to_le_bytes());
        for &value in &self.expansion_weights_q15 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = fnv64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < MAGIC.len() + 8 + 4 + 8 + 8 || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid low-rank artifact".into());
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(bytes[checksum_offset..].try_into()?);
        if fnv64(&bytes[..checksum_offset]) != expected_checksum {
            return Err("low-rank expert checksum mismatch".into());
        }
        let mut offset = MAGIC.len();
        let trunk_model_hash = read_u64(bytes, &mut offset)?;
        let rank = read_u32(bytes, &mut offset)? as usize;
        let projection_seed = read_u64(bytes, &mut offset)?;
        let count = MINI_TRANSFORMER_D_MODEL
            .checked_mul(rank)
            .ok_or("low-rank parameter overflow")?;
        if offset + count * 2 != checksum_offset {
            return Err("low-rank artifact length mismatch".into());
        }
        let mut expansion_weights_q15 = Vec::with_capacity(count);
        for _ in 0..count {
            expansion_weights_q15.push(read_i16(bytes, &mut offset)?);
        }
        let expert = Self {
            trunk_model_hash,
            rank,
            projection_seed,
            expansion_weights_q15,
        };
        expert.validate()?;
        Ok(expert)
    }
}

fn projection_sign(seed: u64, rank_index: usize, dim: usize) -> i64 {
    let mut value = seed
        ^ (rank_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (dim as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    if value & 1 == 0 { 1 } else { -1 }
}

fn latent_q15(hidden: &[i16; MINI_TRANSFORMER_D_MODEL], rank: usize, seed: u64) -> Vec<i16> {
    (0..rank)
        .map(|rank_index| {
            let sum = hidden
                .iter()
                .enumerate()
                .map(|(dim, &value)| i64::from(value) * projection_sign(seed, rank_index, dim))
                .sum::<i64>();
            saturate_i16(round_shift_rhu_i64(sum, 7))
        })
        .collect()
}

fn adapted_hidden(
    base: &[i16; MINI_TRANSFORMER_D_MODEL],
    expert: &LowRankExpert,
) -> ([i16; MINI_TRANSFORMER_D_MODEL], Vec<i16>, usize) {
    let latent = latent_q15(base, expert.rank, expert.projection_seed);
    let mut output = [0_i16; MINI_TRANSFORMER_D_MODEL];
    let mut saturation_count = 0_usize;
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        let residual_acc = (0..expert.rank)
            .map(|rank_index| {
                i64::from(latent[rank_index])
                    * i64::from(expert.expansion_weights_q15[dim * expert.rank + rank_index])
            })
            .sum::<i64>();
        let residual = round_shift_rhu_i64(residual_acc, 15);
        let raw = i64::from(base[dim]).saturating_add(residual);
        output[dim] = saturate_i16(raw);
        saturation_count += usize::from(i64::from(output[dim]) != raw);
    }
    (output, latent, saturation_count)
}

fn evaluate_expert(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &LowRankExpert,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    for record in records {
        let (hidden, _, _) = adapted_hidden(&record.last_hidden_q15, expert);
        let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
        let target = tokens[record.end];
        mistakes += usize::from(argmax(&row.logits_q8) != target);
        probability_error_q15 = probability_error_q15
            .saturating_add(sample_probability_error(&row.probabilities_q15, target));
    }
    Ok(Metrics {
        windows: records.len(),
        mistakes,
        probability_error_q15,
    })
}

#[allow(clippy::too_many_arguments)]
fn train_expert(
    records: &[MiniTransformerMlpWindowEvalRecord],
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    expert: &mut LowRankExpert,
    epochs: usize,
    batch_windows: usize,
    learning_rate: i64,
    learning_rate_shift: u8,
) -> Result<TrainStats, Box<dyn std::error::Error>> {
    let mut residuals = vec![0_i64; expert.expansion_weights_q15.len()];
    let mut stats = TrainStats {
        optimizer_steps: 0,
        weight_delta_l1: 0,
        weight_saturation_count: 0,
        hidden_saturation_count: 0,
    };
    for _ in 0..epochs {
        for batch in records.chunks(batch_windows) {
            let mut gradients = vec![0_i64; expert.expansion_weights_q15.len()];
            for record in batch {
                let (hidden, latent, hidden_saturations) =
                    adapted_hidden(&record.last_hidden_q15, expert);
                stats.hidden_saturation_count = stats
                    .hidden_saturation_count
                    .saturating_add(hidden_saturations);
                let row = mini_transformer_output_from_hidden_q15(model, &hidden)?;
                let target = usize::from(tokens[record.end]);
                let target_probability = i64::from(row.probabilities_q15[target].max(0));
                let mut grad_output_q15 = [0_i16; VOCAB];
                for index in 0..VOCAB {
                    let mut gradient = i64::from(row.probabilities_q15[index]);
                    if index == target {
                        gradient = gradient.saturating_sub(i64::from(i16::MAX));
                    }
                    gradient = gradient.saturating_mul(target_probability) / i64::from(i16::MAX);
                    grad_output_q15[index] = saturate_i16(gradient);
                }
                let grad_hidden_q15 =
                    mini_transformer_output_gradient_to_hidden_q15(model, &grad_output_q15)?;
                for dim in 0..MINI_TRANSFORMER_D_MODEL {
                    if hidden[dim] == i16::MIN || hidden[dim] == i16::MAX {
                        continue;
                    }
                    for rank_index in 0..expert.rank {
                        let index = dim * expert.rank + rank_index;
                        let gradient = round_shift_rhu_i64(
                            i64::from(grad_hidden_q15[dim])
                                .saturating_mul(i64::from(latent[rank_index])),
                            15,
                        );
                        gradients[index] = gradients[index].saturating_add(gradient);
                    }
                }
            }
            let denominator = i64::try_from(batch.len())?
                .checked_shl(u32::from(learning_rate_shift))
                .ok_or("low-rank denominator overflow")?;
            for index in 0..expert.expansion_weights_q15.len() {
                let numerator = gradients[index].saturating_add(residuals[index]);
                let update = round_div_signed(numerator, denominator)?;
                residuals[index] = numerator.saturating_sub(update.saturating_mul(denominator));
                let update = update.saturating_mul(learning_rate);
                let previous = expert.expansion_weights_q15[index];
                let raw = i64::from(previous).saturating_sub(update);
                let next = saturate_i16(raw);
                if i64::from(next) != raw {
                    stats.weight_saturation_count += 1;
                    residuals[index] = 0;
                }
                stats.weight_delta_l1 = stats
                    .weight_delta_l1
                    .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
                expert.expansion_weights_q15[index] = next;
            }
            stats.optimizer_steps += 1;
        }
    }
    Ok(stats)
}

fn sample_probability_error(probabilities: &[i16; VOCAB], target: u8) -> usize {
    let target = usize::from(target);
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities[target])).max(0) as usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if index != target {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
}

fn argmax(values: &[i32; VOCAB]) -> u8 {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map_or(0, |(index, _)| index as u8)
}

fn metrics_json(metrics: Metrics) -> String {
    format!(
        "{{\"windows\":{},\"mistakes\":{},\"accuracy_per_mille\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{}}}",
        metrics.windows,
        metrics.mistakes,
        metrics.windows.saturating_sub(metrics.mistakes) * 1000 / metrics.windows.max(1),
        metrics.probability_error_q15,
        metrics.probability_error_q15 / metrics.windows.max(1),
    )
}

fn round_div_signed(value: i64, denominator: i64) -> Result<i64, &'static str> {
    if denominator <= 0 {
        return Err("invalid signed division denominator");
    }
    let half = denominator / 2;
    Ok(if value >= 0 {
        value.saturating_add(half) / denominator
    } else {
        value.saturating_sub(half) / denominator
    })
}

fn optional_usize_json(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = offset.checked_add(4).ok_or("offset overflow")?;
    let value = u32::from_le_bytes(bytes.get(*offset..end).ok_or("truncated u32")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, Box<dyn std::error::Error>> {
    let end = offset.checked_add(8).ok_or("offset overflow")?;
    let value = u64::from_le_bytes(bytes.get(*offset..end).ok_or("truncated u64")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn read_i16(bytes: &[u8], offset: &mut usize) -> Result<i16, Box<dyn std::error::Error>> {
    let end = offset.checked_add(2).ok_or("offset overflow")?;
    let value = i16::from_le_bytes(bytes.get(*offset..end).ok_or("truncated i16")?.try_into()?);
    *offset = end;
    Ok(value)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_expansion_is_identity() {
        let expert = LowRankExpert::new(1, 4, 7).expect("expert");
        let hidden = core::array::from_fn(|index| index as i16 - 64);
        let (adapted, latent, saturations) = adapted_hidden(&hidden, &expert);
        assert_eq!(adapted, hidden);
        assert_eq!(latent.len(), 4);
        assert_eq!(saturations, 0);
    }

    #[test]
    fn artifact_round_trip_and_corruption_are_locked() {
        let mut expert = LowRankExpert::new(9, 4, 11).expect("expert");
        expert.expansion_weights_q15[5] = -321;
        let bytes = expert.to_bytes().expect("bytes");
        assert_eq!(LowRankExpert::from_bytes(&bytes).expect("decode"), expert);
        let mut corrupt = bytes;
        corrupt[30] ^= 1;
        assert!(LowRankExpert::from_bytes(&corrupt).is_err());
    }
}
