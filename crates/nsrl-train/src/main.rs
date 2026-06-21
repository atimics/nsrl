#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use nsrl_train::{
    ByteDecodePriors, ByteEmbedSoftmaxModel, ByteEmbedSoftmaxTrainConfig, ByteGenerationConfig,
    ByteSoftmaxModel, ByteSoftmaxTrainConfig, ByteTokenizerId, DecodeStrategy, LexemeDecodePriors,
    LexemeEmbeddingModel, LexemeEmbeddingTrainConfig, LexemeGenerationConfig,
    LexemeQualityWeightProfile, LexemeSoftmaxModel, LexemeSoftmaxTrainConfig, LinearBackwardConfig,
    MiniTransformerMlpModel, MiniTransformerMlpTrainConfig, SoftmaxTrainConfig, TrainConfig,
    generate_byte_embed_softmax_with_priors, generate_byte_softmax_with_priors,
    generate_lexeme_softmax_with_priors, generate_mini_transformer_with_priors,
    lexeme_quality_weights_from_vocab, run_byte_embed_softmax_training_with_model,
    run_byte_softmax_training_with_model, run_gated_mlp_backward_smoke,
    run_lexeme_embedding_training_with_model_and_quality,
    run_lexeme_softmax_training_with_model_and_quality, run_linear_backward_smoke,
    run_mini_transformer_mlp_training_from_model, run_mini_transformer_mlp_training_with_model,
    run_softmax_training, run_training_smoke,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = TrainConfig::default();
    let mut softmax_config = SoftmaxTrainConfig::default();
    let mut linear_backward_config = LinearBackwardConfig::default();
    let mut byte_softmax_config = ByteSoftmaxTrainConfig::default();
    let mut byte_embed_softmax_config = ByteEmbedSoftmaxTrainConfig::default();
    let mut lexeme_embedding_config = LexemeEmbeddingTrainConfig::default();
    let mut lexeme_softmax_config = LexemeSoftmaxTrainConfig::default();
    let mut mini_transformer_config = MiniTransformerMlpTrainConfig::default();
    let mut byte_generation_config = ByteGenerationConfig::greedy(32);
    let mut lexeme_generation_config = LexemeGenerationConfig::greedy(32);
    let mut mode = String::from("softmax");
    let mut tokens_path = None;
    let mut model_path = None;
    let mut model_out_path = None;
    let mut vocab_path = None;
    let mut prompt = Vec::new();
    let mut trace_path = None;
    let mut text_out_path = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                mode = args.next().ok_or(
                    "--mode requires softmax, perceptron, linear-backward, gated-mlp-backward, byte-softmax, byte-generate, byte-embed-softmax, byte-embed-generate, mini-transformer-mlp, or mini-transformer-generate",
                )?;
            }
            "--epochs" => {
                let epochs = args
                    .next()
                    .ok_or("--epochs requires a following integer")?
                    .parse()?;
                config.epochs = epochs;
                softmax_config.epochs = epochs;
                byte_softmax_config.epochs = epochs;
                byte_embed_softmax_config.epochs = epochs;
                lexeme_embedding_config.epochs = epochs;
                lexeme_softmax_config.epochs = epochs;
                mini_transformer_config.epochs = epochs;
            }
            "--learning-rate" => {
                let value: i32 = args
                    .next()
                    .ok_or("--learning-rate requires a following integer")?
                    .parse()?;
                config.learning_rate = i8::try_from(value)?;
                softmax_config.learning_rate = value;
                linear_backward_config.learning_rate = value;
                byte_softmax_config.learning_rate = value;
                byte_embed_softmax_config.learning_rate = value;
                lexeme_embedding_config.learning_rate = value;
                lexeme_softmax_config.learning_rate = value;
                mini_transformer_config.learning_rate = value;
            }
            "--lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--lr-shift requires a following integer")?
                    .parse()?;
                softmax_config.learning_rate_shift = value;
                linear_backward_config.learning_rate_shift = value;
                byte_softmax_config.learning_rate_shift = value;
                byte_embed_softmax_config.head_learning_rate_shift = value;
                lexeme_embedding_config.learning_rate_shift = value;
                lexeme_softmax_config.learning_rate_shift = value;
                if lexeme_softmax_config.max_learning_rate_shift < value {
                    lexeme_softmax_config.max_learning_rate_shift = value;
                }
                mini_transformer_config.output_learning_rate_shift = value;
            }
            "--lr-shift-decay-windows" => {
                lexeme_softmax_config.lr_shift_decay_windows = args
                    .next()
                    .ok_or("--lr-shift-decay-windows requires an integer")?
                    .parse()?;
            }
            "--lr-shift-decay-step" => {
                lexeme_softmax_config.lr_shift_decay_step = args
                    .next()
                    .ok_or("--lr-shift-decay-step requires an integer")?
                    .parse()?;
            }
            "--max-lr-shift" => {
                lexeme_softmax_config.max_learning_rate_shift = args
                    .next()
                    .ok_or("--max-lr-shift requires an integer")?
                    .parse()?;
            }
            "--context-radius" => {
                lexeme_embedding_config.context_radius = args
                    .next()
                    .ok_or("--context-radius requires an integer")?
                    .parse()?;
            }
            "--vocab-size" => {
                lexeme_embedding_config.vocab_size = args
                    .next()
                    .ok_or("--vocab-size requires an integer")?
                    .parse()?;
            }
            "--embedding-dim" => {
                lexeme_embedding_config.embedding_dim = args
                    .next()
                    .ok_or("--embedding-dim requires an integer")?
                    .parse()?;
            }
            "--embed-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--embed-lr-shift requires an integer")?
                    .parse()?;
                byte_embed_softmax_config.embedding_learning_rate_shift = value;
                mini_transformer_config.embedding_learning_rate_shift = value;
            }
            "--mlp-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--mlp-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_config.mlp_learning_rate_shift = value;
            }
            "--attention-lr-shift" => {
                mini_transformer_config.attention_learning_rate_shift = args
                    .next()
                    .ok_or("--attention-lr-shift requires an integer")?
                    .parse()?;
            }
            "--attention-qk-lr-shift" => {
                mini_transformer_config.attention_qk_learning_rate_shift = args
                    .next()
                    .ok_or("--attention-qk-lr-shift requires an integer")?
                    .parse()?;
            }
            "--attention-vo-error-feedback" => {
                mini_transformer_config.attention_vo_error_feedback = true;
            }
            "--attention-vo-oracle" => {
                mini_transformer_config.attention_vo_oracle = true;
            }
            "--reject-loss-regression" => {
                mini_transformer_config.reject_loss_regression = true;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(
                    args.next().ok_or("--tokens requires a following path")?,
                ));
            }
            "--model" => {
                model_path = Some(PathBuf::from(
                    args.next().ok_or("--model requires a following path")?,
                ));
            }
            "--model-out" => {
                model_out_path = Some(PathBuf::from(
                    args.next().ok_or("--model-out requires a following path")?,
                ));
            }
            "--vocab" => {
                vocab_path = Some(PathBuf::from(
                    args.next().ok_or("--vocab requires a following path")?,
                ));
            }
            "--prompt" => {
                prompt = args
                    .next()
                    .ok_or("--prompt requires a following string")?
                    .into_bytes();
            }
            "--max-new-tokens" => {
                let value = args
                    .next()
                    .ok_or("--max-new-tokens requires an integer")?
                    .parse()?;
                byte_generation_config.max_new_tokens = value;
                lexeme_generation_config.max_new_tokens = value;
            }
            "--decode" => {
                let value = args.next().ok_or("--decode requires greedy or sample")?;
                let strategy = match value.as_str() {
                    "greedy" => DecodeStrategy::Greedy,
                    "sample" => DecodeStrategy::Sample,
                    _ => return Err("--decode requires greedy or sample".into()),
                };
                byte_generation_config.decode.strategy = strategy;
                lexeme_generation_config.decode.strategy = strategy;
            }
            "--sample-seed" => {
                let value = args
                    .next()
                    .ok_or("--sample-seed requires an integer")?
                    .parse()?;
                byte_generation_config.decode.sample_seed = value;
                lexeme_generation_config.decode.sample_seed = value;
            }
            "--top-k" => {
                let value = args.next().ok_or("--top-k requires an integer")?.parse()?;
                byte_generation_config.decode.top_k = value;
                lexeme_generation_config.decode.top_k = value;
            }
            "--tokenizer" => {
                let value = args
                    .next()
                    .ok_or("--tokenizer requires identity or ascii-lower")?;
                let tokenizer_id = match value.as_str() {
                    "identity" | "byte_identity_u8_v1" => ByteTokenizerId::Identity,
                    "ascii-lower" | "byte_ascii_lower_text_u8_v1" => {
                        ByteTokenizerId::AsciiLowerText
                    }
                    _ => return Err("--tokenizer requires identity or ascii-lower".into()),
                };
                byte_softmax_config.tokenizer_id = tokenizer_id;
                byte_embed_softmax_config.tokenizer_id = tokenizer_id;
                mini_transformer_config.tokenizer_id = tokenizer_id;
                byte_generation_config.tokenizer_id = tokenizer_id;
            }
            "--printable-only" => {
                byte_generation_config.decode.printable_only = true;
            }
            "--ascii-lower-only" => {
                byte_generation_config.decode.ascii_lower_only = true;
            }
            "--repeat-window" => {
                let value = args
                    .next()
                    .ok_or("--repeat-window requires an integer")?
                    .parse()?;
                byte_generation_config.decode.repeat_window = value;
                lexeme_generation_config.decode.repeat_window = value;
            }
            "--repeat-penalty-shift" => {
                let value = args
                    .next()
                    .ok_or("--repeat-penalty-shift requires an integer")?
                    .parse()?;
                byte_generation_config.decode.repeat_penalty_shift = value;
                lexeme_generation_config.decode.repeat_penalty_shift = value;
            }
            "--max-repeat-run" => {
                let value = args
                    .next()
                    .ok_or("--max-repeat-run requires an integer")?
                    .parse()?;
                byte_generation_config.decode.max_repeat_run = value;
                lexeme_generation_config.decode.max_repeat_run = value;
            }
            "--max-weight-delta" => {
                lexeme_softmax_config.max_weight_delta = args
                    .next()
                    .ok_or("--max-weight-delta requires an integer")?
                    .parse()?;
            }
            "--concept-frequency-cap" => {
                lexeme_embedding_config.concept_frequency_cap = args
                    .next()
                    .ok_or("--concept-frequency-cap requires an integer")?
                    .parse()?;
            }
            "--target-frequency-cap" => {
                lexeme_softmax_config.target_frequency_cap = args
                    .next()
                    .ok_or("--target-frequency-cap requires an integer")?
                    .parse()?;
            }
            "--frequency-weight-min-q15" => {
                let value = args
                    .next()
                    .ok_or("--frequency-weight-min-q15 requires an integer")?
                    .parse()?;
                lexeme_embedding_config.concept_frequency_min_weight_q15 = value;
                lexeme_softmax_config.target_frequency_min_weight_q15 = value;
            }
            "--quality-weight-profile" => {
                let value = args
                    .next()
                    .ok_or("--quality-weight-profile requires off or cruft-aware")?;
                let profile = match value.as_str() {
                    "off" => LexemeQualityWeightProfile::Off,
                    "cruft-aware" | "cruft" => LexemeQualityWeightProfile::CruftAware,
                    _ => return Err("--quality-weight-profile requires off or cruft-aware".into()),
                };
                lexeme_embedding_config.quality_weight_profile = profile;
                lexeme_softmax_config.quality_weight_profile = profile;
            }
            "--corpus-prior" => {
                byte_generation_config.decode.corpus_prior = true;
                lexeme_generation_config.decode.corpus_prior = true;
            }
            "--corpus-prior-logit-shift" => {
                let value = args
                    .next()
                    .ok_or("--corpus-prior-logit-shift requires an integer")?
                    .parse()?;
                byte_generation_config.decode.corpus_prior_logit_shift = value;
                lexeme_generation_config.decode.corpus_prior_logit_shift = value;
            }
            "--strict-adjacency" => {
                byte_generation_config.decode.strict_adjacency = true;
                lexeme_generation_config.decode.strict_adjacency = true;
            }
            "--seq-len" => {
                byte_softmax_config.seq_len = args
                    .next()
                    .ok_or("--seq-len requires an integer")?
                    .parse()?;
                byte_embed_softmax_config.seq_len = byte_softmax_config.seq_len;
                lexeme_softmax_config.seq_len = byte_softmax_config.seq_len;
                mini_transformer_config.seq_len = byte_softmax_config.seq_len;
            }
            "--stride" => {
                byte_softmax_config.stride =
                    args.next().ok_or("--stride requires an integer")?.parse()?;
                byte_embed_softmax_config.stride = byte_softmax_config.stride;
                lexeme_embedding_config.stride = byte_softmax_config.stride;
                lexeme_softmax_config.stride = byte_softmax_config.stride;
                mini_transformer_config.stride = byte_softmax_config.stride;
            }
            "--window-offset" => {
                byte_softmax_config.window_offset = args
                    .next()
                    .ok_or("--window-offset requires an integer")?
                    .parse()?;
                byte_embed_softmax_config.window_offset = byte_softmax_config.window_offset;
                lexeme_embedding_config.window_offset = byte_softmax_config.window_offset;
                lexeme_softmax_config.window_offset = byte_softmax_config.window_offset;
                mini_transformer_config.window_offset = byte_softmax_config.window_offset;
            }
            "--batch-windows" => {
                mini_transformer_config.batch_windows = args
                    .next()
                    .ok_or("--batch-windows requires an integer")?
                    .parse()?;
            }
            "--max-windows" => {
                byte_softmax_config.max_windows = Some(
                    args.next()
                        .ok_or("--max-windows requires an integer")?
                        .parse()?,
                );
                byte_embed_softmax_config.max_windows = byte_softmax_config.max_windows;
                lexeme_embedding_config.max_windows = byte_softmax_config.max_windows;
                lexeme_softmax_config.max_windows = byte_softmax_config.max_windows;
                mini_transformer_config.max_windows = byte_softmax_config.max_windows;
            }
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next().ok_or("--trace requires a following path")?,
                ));
            }
            "--text-out" => {
                text_out_path = Some(PathBuf::from(
                    args.next().ok_or("--text-out requires a following path")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let line = match mode.as_str() {
        "perceptron" | "smoke" => run_training_smoke(config)?.to_json_line(),
        "softmax" => run_softmax_training(softmax_config)?.to_json_line(),
        "linear-backward" | "linear_backward" => {
            run_linear_backward_smoke(linear_backward_config)?.to_json_line()
        }
        "gated-mlp-backward" | "gated_mlp_backward" => {
            run_gated_mlp_backward_smoke(linear_backward_config)?.to_json_line()
        }
        "byte-softmax" | "byte_softmax" => {
            let path = tokens_path.ok_or("--tokens is required for byte-softmax mode")?;
            let tokens = fs::read(path)?;
            let run = run_byte_softmax_training_with_model(&tokens, byte_softmax_config)?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "byte-generate" | "byte_generate" => {
            let path = model_path.ok_or("--model is required for byte-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for byte-generate mode".into());
            }
            let model_bytes = fs::read(path)?;
            let model = ByteSoftmaxModel::from_bytes(&model_bytes)?;
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let generation = generate_byte_softmax_with_priors(
                &model,
                &prompt,
                byte_generation_config,
                decode_priors.as_ref(),
            )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
            )?;
            generation.to_json_line()
        }
        "byte-embed-softmax" | "byte_embed_softmax" => {
            let path = tokens_path.ok_or("--tokens is required for byte-embed-softmax mode")?;
            let tokens = fs::read(path)?;
            let run =
                run_byte_embed_softmax_training_with_model(&tokens, byte_embed_softmax_config)?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "byte-embed-generate" | "byte_embed_generate" => {
            let path = model_path.ok_or("--model is required for byte-embed-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for byte-embed-generate mode".into());
            }
            let model_bytes = fs::read(path)?;
            let model = ByteEmbedSoftmaxModel::from_bytes(&model_bytes)?;
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let generation = generate_byte_embed_softmax_with_priors(
                &model,
                &prompt,
                byte_generation_config,
                decode_priors.as_ref(),
            )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
            )?;
            generation.to_json_line()
        }
        "lexeme-embedding" | "lexeme_embedding" => {
            let path = tokens_path.ok_or("--tokens is required for lexeme-embedding mode")?;
            let tokens = fs::read(path)?;
            let quality_weights = load_lexeme_quality_weights(
                &vocab_path,
                lexeme_embedding_config.vocab_size,
                lexeme_embedding_config.quality_weight_profile,
            )?;
            let run = run_lexeme_embedding_training_with_model_and_quality(
                &tokens,
                lexeme_embedding_config,
                quality_weights.as_deref(),
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "lexeme-softmax" | "lexeme_softmax" => {
            let token_path = tokens_path.ok_or("--tokens is required for lexeme-softmax mode")?;
            let model_path = model_path.ok_or("--model is required for lexeme-softmax mode")?;
            let tokens = fs::read(token_path)?;
            let embedding_model = LexemeEmbeddingModel::from_bytes(&fs::read(model_path)?)?;
            let quality_weights = load_lexeme_quality_weights(
                &vocab_path,
                embedding_model.vocab_size,
                lexeme_softmax_config.quality_weight_profile,
            )?;
            let run = run_lexeme_softmax_training_with_model_and_quality(
                &tokens,
                embedding_model,
                lexeme_softmax_config,
                quality_weights.as_deref(),
            )?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "lexeme-generate" | "lexeme_generate" => {
            let model_path = model_path.ok_or("--model is required for lexeme-generate mode")?;
            let vocab_path = vocab_path.ok_or("--vocab is required for lexeme-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for lexeme-generate mode".into());
            }
            let vocab = load_lexeme_vocab(&vocab_path)?;
            let model = LexemeSoftmaxModel::from_bytes(&fs::read(model_path)?)?;
            let prompt_tokens = lexeme_prompt_tokens(&prompt, &vocab.lookup);
            let decode_priors = load_lexeme_decode_priors(
                &tokens_path,
                model.vocab_size,
                lexeme_generation_config,
            )?;
            let generation = generate_lexeme_softmax_with_priors(
                &model,
                &prompt_tokens,
                lexeme_generation_config,
                decode_priors.as_ref(),
            )?;
            write_lexeme_text_generation(
                &text_out_path,
                &generation.prompt_tokens,
                &generation.generated_tokens,
                &vocab.entries,
            )?;
            generation.to_json_line()
        }
        "mini-transformer-mlp" | "mini_transformer_mlp" => {
            let path = tokens_path.ok_or("--tokens is required for mini-transformer-mlp mode")?;
            let tokens = fs::read(path)?;
            let run = if let Some(path) = model_path {
                let model_bytes = fs::read(path)?;
                let model = MiniTransformerMlpModel::from_bytes(&model_bytes)?;
                run_mini_transformer_mlp_training_from_model(
                    &tokens,
                    mini_transformer_config,
                    model,
                )?
            } else {
                run_mini_transformer_mlp_training_with_model(&tokens, mini_transformer_config)?
            };
            if let Some(path) = model_out_path {
                fs::write(path, run.model.to_bytes())?;
            }
            run.trace.to_json_line()
        }
        "mini-transformer-generate" | "mini_transformer_generate" => {
            let path =
                model_path.ok_or("--model is required for mini-transformer-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for mini-transformer-generate mode".into());
            }
            let model_bytes = fs::read(path)?;
            let model = MiniTransformerMlpModel::from_bytes(&model_bytes)?;
            let decode_priors = load_decode_priors(&tokens_path, byte_generation_config)?;
            let generation = generate_mini_transformer_with_priors(
                &model,
                &prompt,
                byte_generation_config,
                decode_priors.as_ref(),
            )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
            )?;
            generation.to_json_line()
        }
        other => return Err(format!("unknown mode: {other}").into()),
    };
    if let Some(path) = trace_path {
        fs::write(path, line)?;
    } else {
        io::stdout().write_all(line.as_bytes())?;
    }

    Ok(())
}

fn print_help() {
    println!(
        "Usage: nsrl-train [--mode softmax|perceptron|linear-backward|gated-mlp-backward|byte-softmax|byte-generate|byte-embed-softmax|byte-embed-generate|lexeme-embedding|lexeme-softmax|lexeme-generate|mini-transformer-mlp|mini-transformer-generate] [--tokens PATH] [--model PATH] [--model-out PATH] [--vocab PATH] [--prompt TEXT] [--max-new-tokens N] [--decode greedy|sample] [--sample-seed N] [--top-k N] [--tokenizer identity|ascii-lower] [--printable-only] [--ascii-lower-only] [--repeat-window N] [--repeat-penalty-shift N] [--max-repeat-run N] [--corpus-prior] [--corpus-prior-logit-shift N] [--strict-adjacency] [--epochs N] [--learning-rate N] [--lr-shift N] [--lr-shift-decay-windows N] [--lr-shift-decay-step N] [--max-lr-shift N] [--max-weight-delta N] [--concept-frequency-cap N] [--target-frequency-cap N] [--frequency-weight-min-q15 N] [--quality-weight-profile off|cruft-aware] [--context-radius N] [--vocab-size N] [--embedding-dim N] [--mlp-lr-shift N] [--embed-lr-shift N] [--attention-lr-shift N] [--attention-qk-lr-shift N] [--attention-vo-error-feedback] [--attention-vo-oracle] [--reject-loss-regression] [--seq-len N] [--stride N] [--window-offset N] [--batch-windows N] [--max-windows N] [--trace PATH] [--text-out PATH]"
    );
    println!();
    println!("Runs a deterministic integer training trace.");
}

struct LexemeVocab {
    entries: Vec<String>,
    lookup: HashMap<String, u16>,
}

fn load_lexeme_vocab(path: &PathBuf) -> Result<LexemeVocab, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut entries = Vec::new();
    let mut lookup = HashMap::new();
    entries.resize(256, String::new());
    for id in 0..256 {
        entries[id] = String::from_utf8_lossy(&[id as u8]).to_string();
    }

    for line in text.lines().skip(1) {
        let mut parts = line.splitn(3, '\t');
        let Some(id) = parts.next() else { continue };
        let Some(lexeme) = parts.next() else { continue };
        let id: usize = id.parse()?;
        if id > usize::from(u16::MAX) {
            return Err("lexeme vocab id exceeds u16".into());
        }
        if entries.len() <= id {
            entries.resize(id + 1, String::new());
        }
        entries[id] = lexeme.to_string();
        lookup.insert(lexeme.to_string(), id as u16);
    }

    Ok(LexemeVocab { entries, lookup })
}

fn load_lexeme_quality_weights(
    vocab_path: &Option<PathBuf>,
    vocab_size: usize,
    profile: LexemeQualityWeightProfile,
) -> Result<Option<Vec<i16>>, Box<dyn std::error::Error>> {
    if profile == LexemeQualityWeightProfile::Off {
        return Ok(None);
    }
    let path = vocab_path
        .as_ref()
        .ok_or("--vocab is required with --quality-weight-profile cruft-aware")?;
    let vocab = load_lexeme_vocab(path)?;
    Ok(Some(lexeme_quality_weights_from_vocab(
        &vocab.entries,
        vocab_size,
        profile,
    )?))
}

fn lexeme_prompt_tokens(input: &[u8], lookup: &HashMap<String, u16>) -> Vec<u16> {
    let normalized = input
        .iter()
        .map(|&byte| {
            if byte.is_ascii_uppercase() {
                byte.to_ascii_lowercase()
            } else {
                byte
            }
        })
        .collect::<Vec<_>>();
    let mut tokens = Vec::new();
    for_each_prompt_lexeme(&normalized, |lexeme| {
        let text = String::from_utf8_lossy(lexeme).to_string();
        if let Some(&id) = lookup.get(&text) {
            tokens.push(id);
        } else {
            tokens.extend(lexeme.iter().map(|&byte| u16::from(byte)));
        }
    });
    if tokens.is_empty() {
        tokens.push(u16::from(b' '));
    }
    tokens
}

fn for_each_prompt_lexeme<F>(input: &[u8], mut visit: F)
where
    F: FnMut(&[u8]),
{
    let mut index = 0_usize;
    while index < input.len() {
        let byte = input[index];
        if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if is_prompt_word_byte(byte) {
            let start = index;
            index += 1;
            while index < input.len() {
                let current = input[index];
                if is_prompt_word_byte(current)
                    || ((current == b'\'' || current == b'-')
                        && index + 1 < input.len()
                        && is_prompt_word_byte(input[index - 1])
                        && is_prompt_word_byte(input[index + 1]))
                {
                    index += 1;
                } else {
                    break;
                }
            }
            visit(&input[start..index]);
            continue;
        }
        if is_prompt_punctuation(byte) {
            visit(&input[index..index + 1]);
        }
        index += 1;
    }
}

fn is_prompt_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

fn is_prompt_punctuation(byte: u8) -> bool {
    matches!(byte, b'.' | b',' | b';' | b':' | b'?' | b'!' | b'\'' | b'-')
}

fn write_lexeme_text_generation(
    path: &Option<PathBuf>,
    prompt_tokens: &[u16],
    generated_tokens: &[u16],
    vocab: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let mut all_tokens = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        all_tokens.extend_from_slice(prompt_tokens);
        all_tokens.extend_from_slice(generated_tokens);
        fs::write(path, render_lexeme_tokens(&all_tokens, vocab))?;
    }
    Ok(())
}

fn render_lexeme_tokens(tokens: &[u16], vocab: &[String]) -> String {
    let mut out = String::new();
    for &token in tokens {
        let text = vocab
            .get(usize::from(token))
            .filter(|value| !value.is_empty())
            .cloned()
            .unwrap_or_else(|| {
                if token <= u16::from(u8::MAX) {
                    String::from_utf8_lossy(&[token as u8]).to_string()
                } else {
                    String::from("?")
                }
            });
        append_rendered_lexeme(&mut out, &text);
    }
    out
}

fn append_rendered_lexeme(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if text == " " {
        if !out.ends_with(' ') {
            out.push(' ');
        }
        return;
    }
    if is_render_punctuation(text) {
        while out.ends_with(' ') {
            out.pop();
        }
        out.push_str(text);
        return;
    }
    if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\'') && !out.ends_with('-') {
        out.push(' ');
    }
    out.push_str(text);
}

fn is_render_punctuation(text: &str) -> bool {
    matches!(text, "." | "," | ";" | ":" | "?" | "!" | "'" | "-")
}

fn load_decode_priors(
    tokens_path: &Option<PathBuf>,
    config: ByteGenerationConfig,
) -> Result<Option<ByteDecodePriors>, Box<dyn std::error::Error>> {
    if !config.decode.corpus_prior && !config.decode.strict_adjacency {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with --corpus-prior or --strict-adjacency")?;
    let tokens = fs::read(path)?;
    Ok(Some(ByteDecodePriors::from_tokens(&tokens)?))
}

fn load_lexeme_decode_priors(
    tokens_path: &Option<PathBuf>,
    vocab_size: usize,
    config: LexemeGenerationConfig,
) -> Result<Option<LexemeDecodePriors>, Box<dyn std::error::Error>> {
    if !config.decode.corpus_prior && !config.decode.strict_adjacency {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with --corpus-prior or --strict-adjacency")?;
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err("lexeme token stream must contain little-endian u16 tokens".into());
    }
    let tokens = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    Ok(Some(LexemeDecodePriors::from_tokens(&tokens, vocab_size)?))
}

fn write_text_generation(
    path: &Option<PathBuf>,
    prompt: &[u8],
    generated: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let mut text = Vec::with_capacity(prompt.len() + generated.len());
        text.extend_from_slice(prompt);
        text.extend_from_slice(generated);
        fs::write(path, text)?;
    }
    Ok(())
}
