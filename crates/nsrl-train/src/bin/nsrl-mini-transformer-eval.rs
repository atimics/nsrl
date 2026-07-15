#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerPositionPolicy, evaluate_mini_transformer_mlp_model,
    evaluate_mini_transformer_mlp_windows,
};

const SUFFIX_MEMORY_MAGIC: [u8; 8] = *b"NSRLSM1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalAblation {
    Combined,
    TransformerOnly,
    SuffixMemoryOnly,
}

impl EvalAblation {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "combined" => Ok(Self::Combined),
            "transformer-only" => Ok(Self::TransformerOnly),
            "suffix-memory-only" => Ok(Self::SuffixMemoryOnly),
            _ => {
                Err("--ablation requires combined, transformer-only, or suffix-memory-only".into())
            }
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Combined => "combined",
            Self::TransformerOnly => "transformer-only",
            Self::SuffixMemoryOnly => "suffix-memory-only",
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens_path = None;
    let mut model_path = None;
    let mut out_path = None;
    let mut ablated_model_out_path = None;
    let mut details_out_path = None;
    let mut logits_out_path = None;
    let mut stride = 1_usize;
    let mut max_windows = None;
    let mut attention_kind = MiniTransformerAttentionKind::Linear;
    let mut position_policy = MiniTransformerPositionPolicy::Nope;
    let mut ablation = EvalAblation::Combined;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens_path = Some(PathBuf::from(required(&mut args, "--tokens")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => out_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--ablated-model-out" => {
                ablated_model_out_path =
                    Some(PathBuf::from(required(&mut args, "--ablated-model-out")?))
            }
            "--details-out" => {
                details_out_path = Some(PathBuf::from(required(&mut args, "--details-out")?))
            }
            "--logits-out" => {
                logits_out_path = Some(PathBuf::from(required(&mut args, "--logits-out")?))
            }
            "--stride" => stride = required(&mut args, "--stride")?.parse()?,
            "--max-windows" => max_windows = Some(required(&mut args, "--max-windows")?.parse()?),
            "--attention" => {
                attention_kind = match required(&mut args, "--attention")?.as_str() {
                    "base2-softmax" => MiniTransformerAttentionKind::Base2Softmax,
                    "linear" => MiniTransformerAttentionKind::Linear,
                    _ => return Err("--attention requires base2-softmax or linear".into()),
                }
            }
            "--position" => {
                position_policy = match required(&mut args, "--position")?.as_str() {
                    "learned-absolute" => MiniTransformerPositionPolicy::LearnedAbsolute,
                    "nope" => MiniTransformerPositionPolicy::Nope,
                    _ => return Err("--position requires learned-absolute or nope".into()),
                }
            }
            "--ablation" => ablation = EvalAblation::parse(&required(&mut args, "--ablation")?)?,
            "--help" | "-h" => {
                println!(
                    "Usage: nsrl-mini-transformer-eval --tokens PATH --model PATH [--out PATH] [--ablated-model-out PATH] [--details-out PATH] [--logits-out PATH] \
                     [--stride N] [--max-windows N] [--attention base2-softmax|linear] \
                     [--position learned-absolute|nope] \
                     [--ablation combined|transformer-only|suffix-memory-only]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let tokens = fs::read(tokens_path.ok_or("--tokens is required")?)?;
    let mut model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let source_model_hash = model.model_hash();
    let source_suffix_memory_present = suffix_memory_is_installed(&model);
    apply_ablation(&mut model, ablation)?;
    if ablated_model_out_path.is_some() && ablation == EvalAblation::Combined {
        return Err("--ablated-model-out requires a non-combined --ablation".into());
    }
    if let Some(path) = ablated_model_out_path {
        fs::write(path, model.try_to_bytes()?)?;
    }
    let evaluated_model_hash = model.model_hash();
    let config = MiniTransformerMlpEvalConfig {
        seq_len: model.context_seq_len,
        stride,
        max_windows,
        attention_kind,
        position_policy,
    };
    let trace = annotate_trace(
        evaluate_mini_transformer_mlp_model(&tokens, &model, config)?.to_json_line(),
        ablation,
        source_model_hash,
        evaluated_model_hash,
        source_suffix_memory_present,
    )?;
    if details_out_path.is_some() || logits_out_path.is_some() {
        let records = evaluate_mini_transformer_mlp_windows(&tokens, &model, config)?;
        let mut details = String::from(
            "start\tend\ttarget\tpredicted\tprevious\ttarget_logit_q8\tpredicted_logit_q8\ttarget_seen_in_context\ttarget_last_distance\tprobability_error_q15\n",
        );
        let mut raw_logits = Vec::with_capacity(records.len().saturating_mul(256 * 4));
        for record in records {
            let target = tokens[record.end];
            let previous = tokens[record.end - 1];
            let predicted = record
                .predicted_token
                .ok_or("evaluation record has no prediction")?;
            let logits = record.logits_q8.ok_or("evaluation record has no logits")?;
            let context = &tokens[record.start..record.end];
            let target_last_distance = context
                .iter()
                .rposition(|&token| token == target)
                .map(|index| context.len() - index)
                .unwrap_or(0);
            details.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                record.start,
                record.end,
                target,
                predicted,
                previous,
                logits[usize::from(target)],
                logits[usize::from(predicted)],
                usize::from(target_last_distance > 0),
                target_last_distance,
                record.probability_error_q15,
            ));
            for logit in logits {
                raw_logits.extend_from_slice(&logit.to_le_bytes());
            }
        }
        if let Some(path) = details_out_path {
            fs::write(path, details)?;
        }
        if let Some(path) = logits_out_path {
            fs::write(path, raw_logits)?;
        }
    }
    if let Some(path) = out_path {
        fs::write(path, trace)?;
    } else {
        print!("{trace}");
    }
    Ok(())
}

fn suffix_memory_is_installed(model: &MiniTransformerMlpModel) -> bool {
    SUFFIX_MEMORY_MAGIC
        .chunks_exact(2)
        .zip(model.position_embeddings.iter())
        .all(|(bytes, &value)| value.to_le_bytes() == bytes)
}

fn apply_ablation(
    model: &mut MiniTransformerMlpModel,
    ablation: EvalAblation,
) -> Result<(), Box<dyn std::error::Error>> {
    match ablation {
        EvalAblation::Combined => Ok(()),
        EvalAblation::TransformerOnly | EvalAblation::SuffixMemoryOnly
            if !suffix_memory_is_installed(model) =>
        {
            Err("requested ablation requires an installed suffix memory".into())
        }
        EvalAblation::TransformerOnly => {
            let header_words = SUFFIX_MEMORY_MAGIC.len() / 2;
            model.position_embeddings[..header_words].fill(0);
            Ok(())
        }
        EvalAblation::SuffixMemoryOnly => {
            model.output_weights.fill(0);
            Ok(())
        }
    }
}

fn annotate_trace(
    trace: String,
    ablation: EvalAblation,
    source_model_hash: u64,
    evaluated_model_hash: u64,
    source_suffix_memory_present: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let marker = "\"model\":{";
    if !trace.contains(marker) {
        return Err("evaluation trace is missing its model object".into());
    }
    let suffix_memory_enabled =
        ablation != EvalAblation::TransformerOnly && source_suffix_memory_present;
    let metadata = format!(
        "\"ablation\":{{\"mode\":\"{}\",\"source_model_hash\":\"0x{:016x}\",\"evaluated_model_hash\":\"0x{:016x}\",\"source_suffix_memory_present\":{},\"suffix_memory_enabled\":{},\"retrieval_enabled\":false,\"routing_oracle_enabled\":false}},{}",
        ablation.as_str(),
        source_model_hash,
        evaluated_model_hash,
        source_suffix_memory_present,
        suffix_memory_enabled,
        marker,
    );
    Ok(trace.replacen(marker, &metadata, 1))
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with_suffix_memory() -> MiniTransformerMlpModel {
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        for (word, bytes) in model
            .position_embeddings
            .iter_mut()
            .zip(SUFFIX_MEMORY_MAGIC.chunks_exact(2))
        {
            *word = i16::from_le_bytes([bytes[0], bytes[1]]);
        }
        model
    }

    #[test]
    fn transformer_only_disables_only_the_suffix_header() {
        let mut model = model_with_suffix_memory();
        let output_weights = model.output_weights.clone();
        apply_ablation(&mut model, EvalAblation::TransformerOnly).expect("ablation");
        assert!(!suffix_memory_is_installed(&model));
        assert_eq!(model.output_weights, output_weights);

        let reloaded = MiniTransformerMlpModel::from_bytes(
            &model
                .try_to_bytes()
                .expect("serialized transformer-only model"),
        )
        .expect("reloaded transformer-only model");
        assert!(!suffix_memory_is_installed(&reloaded));
        assert_eq!(reloaded.model_hash(), model.model_hash());
    }

    #[test]
    fn suffix_only_neutralizes_the_neural_output_head() {
        let mut model = model_with_suffix_memory();
        apply_ablation(&mut model, EvalAblation::SuffixMemoryOnly).expect("ablation");
        assert!(suffix_memory_is_installed(&model));
        assert!(model.output_weights.iter().all(|&weight| weight == 0));
    }

    #[test]
    fn ablations_reject_models_without_suffix_memory() {
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(4);
        assert!(apply_ablation(&mut model, EvalAblation::TransformerOnly).is_err());
        assert!(apply_ablation(&mut model, EvalAblation::SuffixMemoryOnly).is_err());
    }

    #[test]
    fn transformer_only_trace_records_all_assistance_as_disabled() {
        let trace = annotate_trace(
            "{\"model\":{}}\n".to_string(),
            EvalAblation::TransformerOnly,
            1,
            2,
            true,
        )
        .expect("trace");
        assert!(trace.contains("\"source_suffix_memory_present\":true"));
        assert!(trace.contains("\"suffix_memory_enabled\":false"));
        assert!(trace.contains("\"retrieval_enabled\":false"));
        assert!(trace.contains("\"routing_oracle_enabled\":false"));
    }
}
