#![deny(unsafe_code)]

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

use nsrl_core::SoftmaxNormalization;
use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_train::production::{
    ProductionBackwardQuantization, ProductionFullTrainConfig, ProductionModelV1,
    ProductionOptimizerStateV2, decode_bound_token_stream, train_production_full_smoke,
};

const SCHEMA: &str = "nsrl.production_saturation_backoff_audit.v1";
const AUDIT_SOURCE: &[u8] = include_bytes!("nsrl-production-saturation-backoff-audit.rs");
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Config {
    tokenizer: PathBuf,
    tokens: PathBuf,
    model: PathBuf,
    optimizer_state: PathBuf,
    trace: PathBuf,
    output_backward_shifts: Vec<u8>,
    backward_quantization: ProductionBackwardQuantization,
    backward_stochastic_seed: u64,
    candidate_schedule: CandidateSchedule,
}

#[derive(Debug, Clone, Copy)]
struct CandidateSchedule {
    embedding_learning_rate_shift: u8,
    embedding_learning_rate_boost_shift: u8,
    k_learning_rate_shift: u8,
    v_learning_rate_shift: u8,
    o_learning_rate_shift: u8,
    flush_batched_embedding_residuals: bool,
    descent_guard_windows: usize,
    descent_guard_signed_representation_blocks: bool,
}

impl Default for CandidateSchedule {
    fn default() -> Self {
        Self {
            embedding_learning_rate_shift: 0,
            embedding_learning_rate_boost_shift: 2,
            k_learning_rate_shift: 22,
            v_learning_rate_shift: 26,
            o_learning_rate_shift: 10,
            flush_batched_embedding_residuals: false,
            descent_guard_windows: 0,
            descent_guard_signed_representation_blocks: false,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-saturation-backoff-audit: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let token_bytes = fs::read(&config.tokens)?;
    let model_bytes = fs::read(&config.model)?;
    let optimizer_bytes = fs::read(&config.optimizer_state)?;
    let tokenizer = SubwordTokenizer::from_bytes(&tokenizer_bytes)?;
    let model = ProductionModelV1::from_bytes(&model_bytes)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) =
        decode_bound_token_stream(&token_bytes, model.tokenizer_hash, model.config.vocab_size)?;
    let source_state = ProductionOptimizerStateV2::from_bytes(&optimizer_bytes)?;
    let base_config = representation_v2_config(
        8,
        ProductionBackwardQuantization::RescuedRhu,
        0,
        CandidateSchedule::default(),
    );
    source_state.validate_binding(&model, token_stream_hash, base_config)?;

    let mut rows = String::new();
    for (index, &shift) in config.output_backward_shifts.iter().enumerate() {
        let candidate_config = representation_v2_config(
            shift,
            config.backward_quantization,
            config.backward_stochastic_seed,
            config.candidate_schedule,
        );
        let mut candidate_state = source_state.clone();
        candidate_state.schedule_hash =
            ProductionOptimizerStateV2::new(&model, token_stream_hash, candidate_config)
                .schedule_hash;
        let mut candidate_model = model.clone();
        let (trace, _) = train_production_full_smoke(
            &mut candidate_model,
            &tokens,
            token_stream_hash,
            candidate_config,
            Some(candidate_state),
        )?;
        if index > 0 {
            rows.push(',');
        }
        write!(
            rows,
            "{{\"output_backward_shift\":{shift},\"candidate_schedule_hash\":\"0x{:016x}\",\"train_trace\":{}}}",
            ProductionOptimizerStateV2::new(&model, token_stream_hash, candidate_config)
                .schedule_hash,
            trace.to_json_line().trim(),
        )?;
    }

    let mut shifts = String::new();
    for (index, shift) in config.output_backward_shifts.iter().enumerate() {
        if index > 0 {
            shifts.push(',');
        }
        write!(shifts, "{shift}")?;
    }
    let output = format!(
        concat!(
            "{{\"schema\":\"{}\",\"bindings\":{{",
            "\"model_hash\":\"0x{:016x}\",",
            "\"optimizer_state_hash\":\"0x{:016x}\",",
            "\"tokenizer_hash\":\"0x{:016x}\",",
            "\"token_stream_hash\":\"0x{:016x}\",",
            "\"model_artifact_fnv64\":\"0x{:016x}\",",
            "\"optimizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"tokenizer_artifact_fnv64\":\"0x{:016x}\",",
            "\"token_artifact_fnv64\":\"0x{:016x}\",",
            "\"audit_source_fnv64\":\"0x{:016x}\",",
            "\"audit_binary_fnv64\":\"0x{:016x}\"}},",
            "\"source_cursor\":{{\"total_optimizer_step\":{},",
            "\"next_epoch\":{},\"next_window\":{}}},",
            "\"audit\":{{\"mode\":\"read_only_inherited_residual_counterfactual\",",
            "\"base_output_backward_shift\":8,",
            "\"candidate_backward_quantization\":\"{}\",",
            "\"candidate_backward_stochastic_seed\":{},",
            "\"candidate_embedding_learning_rate_shift\":{},",
            "\"candidate_embedding_learning_rate_boost_shift\":{},",
            "\"candidate_k_learning_rate_shift\":{},",
            "\"candidate_v_learning_rate_shift\":{},",
            "\"candidate_o_learning_rate_shift\":{},",
            "\"candidate_flush_batched_embedding_residuals\":{},",
            "\"candidate_descent_guard_windows\":{},",
            "\"candidate_descent_guard_signed_representation_blocks\":{},",
            "\"output_backward_shifts\":[{}],",
            "\"candidate_schedule_hash_rebound_in_memory_only\":true,",
            "\"candidate_artifacts_persisted\":false}},",
            "\"rows\":[{}]}}\n"
        ),
        SCHEMA,
        model.model_hash(),
        source_state.state_hash(),
        model.tokenizer_hash,
        token_stream_hash,
        fnv64(&model_bytes),
        fnv64(&optimizer_bytes),
        fnv64(&tokenizer_bytes),
        fnv64(&token_bytes),
        fnv64(AUDIT_SOURCE),
        fnv64(&fs::read(env::current_exe()?)?),
        source_state.step,
        source_state.next_epoch,
        source_state.next_window,
        config.backward_quantization.as_str(),
        config.backward_stochastic_seed,
        config.candidate_schedule.embedding_learning_rate_shift,
        config
            .candidate_schedule
            .embedding_learning_rate_boost_shift,
        config.candidate_schedule.k_learning_rate_shift,
        config.candidate_schedule.v_learning_rate_shift,
        config.candidate_schedule.o_learning_rate_shift,
        config.candidate_schedule.flush_batched_embedding_residuals,
        config.candidate_schedule.descent_guard_windows,
        config
            .candidate_schedule
            .descent_guard_signed_representation_blocks,
        shifts,
        rows,
    );
    if let Some(parent) = config.trace.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.trace, &output)?;
    print!("{output}");
    Ok(())
}

fn representation_v2_config(
    output_backward_shift: u8,
    backward_quantization: ProductionBackwardQuantization,
    backward_stochastic_seed: u64,
    candidate: CandidateSchedule,
) -> ProductionFullTrainConfig {
    ProductionFullTrainConfig {
        context_tokens: 64,
        max_windows: 2048,
        spread_windows: true,
        targets_per_window: 8,
        training_workers: 8,
        epochs: 1,
        matrix_learning_rate_shift: 59,
        q_learning_rate_shift: Some(59),
        k_learning_rate_shift: Some(candidate.k_learning_rate_shift),
        v_learning_rate_shift: Some(candidate.v_learning_rate_shift),
        o_learning_rate_shift: Some(candidate.o_learning_rate_shift),
        up_learning_rate_shift: Some(59),
        gate_learning_rate_shift: Some(59),
        down_learning_rate_shift: Some(59),
        vector_learning_rate_shift: 62,
        output_bias_learning_rate_shift: Some(51),
        final_rms_learning_rate_shift: Some(59),
        embedding_learning_rate_shift: candidate.embedding_learning_rate_shift,
        embedding_learning_rate_boost_shift: candidate.embedding_learning_rate_boost_shift,
        output_learning_rate_shift: 51,
        output_backward_shift: Some(output_backward_shift),
        probability_gradient_fractional_bits: 23,
        probability_normalization: SoftmaxNormalization::Q47Newton1,
        batch_windows: 4,
        max_optimizer_steps: 1,
        evaluation_windows: 64,
        reject_saturated_batch: true,
        flush_batched_embedding_residuals: candidate.flush_batched_embedding_residuals,
        descent_guard_windows: candidate.descent_guard_windows,
        descent_guard_signed_representation_blocks: candidate
            .descent_guard_signed_representation_blocks,
        backward_quantization,
        backward_stochastic_seed,
    }
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut tokenizer = None;
    let mut tokens = None;
    let mut model = None;
    let mut optimizer_state = None;
    let mut trace = None;
    let mut output_backward_shifts = None;
    let mut backward_quantization = ProductionBackwardQuantization::RescuedRhu;
    let mut backward_stochastic_seed = 0_u64;
    let mut candidate_schedule = CandidateSchedule::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let value = || "missing saturation backoff audit argument value";
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--tokens" => tokens = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--model" => model = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--optimizer-state" => {
                optimizer_state = Some(PathBuf::from(args.next().ok_or_else(value)?))
            }
            "--trace" => trace = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--output-backward-shifts" => {
                output_backward_shifts = Some(parse_shifts(&args.next().ok_or_else(value)?)?)
            }
            "--backward-quantization" => {
                let mode = args.next().ok_or_else(value)?;
                backward_quantization = match mode.as_str() {
                    "rescued-rhu" | "rescued_rhu" => ProductionBackwardQuantization::RescuedRhu,
                    "late-rhu" | "late_rhu" => ProductionBackwardQuantization::LateRhu,
                    "late-stochastic" | "late_stochastic" => {
                        ProductionBackwardQuantization::LateStochastic
                    }
                    _ => return Err(format!("unsupported backward quantization: {mode}").into()),
                };
            }
            "--backward-stochastic-seed" => {
                backward_stochastic_seed = args.next().ok_or_else(value)?.parse()?
            }
            "--embedding-learning-rate-shift" => {
                candidate_schedule.embedding_learning_rate_shift =
                    args.next().ok_or_else(value)?.parse()?
            }
            "--embedding-learning-rate-boost-shift" => {
                candidate_schedule.embedding_learning_rate_boost_shift =
                    args.next().ok_or_else(value)?.parse()?
            }
            "--k-learning-rate-shift" => {
                candidate_schedule.k_learning_rate_shift = args.next().ok_or_else(value)?.parse()?
            }
            "--v-learning-rate-shift" => {
                candidate_schedule.v_learning_rate_shift = args.next().ok_or_else(value)?.parse()?
            }
            "--o-learning-rate-shift" => {
                candidate_schedule.o_learning_rate_shift = args.next().ok_or_else(value)?.parse()?
            }
            "--flush-batched-embedding-residuals" => {
                candidate_schedule.flush_batched_embedding_residuals = true
            }
            "--descent-guard-windows" => {
                candidate_schedule.descent_guard_windows = args.next().ok_or_else(value)?.parse()?
            }
            "--descent-guard-signed-representation-blocks" => {
                candidate_schedule.descent_guard_signed_representation_blocks = true
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        model: model.ok_or("--model is required")?,
        optimizer_state: optimizer_state.ok_or("--optimizer-state is required")?,
        trace: trace.ok_or("--trace is required")?,
        output_backward_shifts: output_backward_shifts
            .ok_or("--output-backward-shifts is required")?,
        backward_quantization,
        backward_stochastic_seed,
        candidate_schedule,
    })
}

fn parse_shifts(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let shifts = value
        .split(',')
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>()?;
    if shifts.is_empty() || shifts.iter().any(|&shift| shift > 30) {
        return Err(
            "output backward shifts must be a nonempty comma-separated list in 0..=30".into(),
        );
    }
    Ok(shifts)
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::parse_shifts;

    #[test]
    fn shift_parser_rejects_empty_and_out_of_range_values() {
        assert_eq!(parse_shifts("8,9,10").unwrap(), [8, 9, 10]);
        assert!(parse_shifts("").is_err());
        assert!(parse_shifts("31").is_err());
    }
}
