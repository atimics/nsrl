#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use nsrl_corpus::LEXEME_PAGE_BOUNDARY;
use nsrl_corpus::encode_lexeme_prompt_tokens;
use nsrl_train::{
    ByteDecodePriors, ByteEmbedSoftmaxModel, ByteEmbedSoftmaxTrainConfig, ByteGenerationConfig,
    ByteSoftmaxModel, ByteSoftmaxTrainConfig, ByteTokenizerId, DecodeStrategy,
    LEXEME_SENTENCE_STOP_TOKEN_CAP, LexemeContextFeatures, LexemeDecodePriors,
    LexemeEmbeddingModel, LexemeEmbeddingTrainConfig, LexemeGenerationConfig, LexemeMemoryPriors,
    LexemeQualityWeightProfile, LexemeSoftmaxModel, LexemeSoftmaxTrainConfig, LexemeTopicPriors,
    LinearBackwardConfig, MiniTransformerAttentionKind, MiniTransformerMlpModel,
    MiniTransformerMlpTrainConfig, MiniTransformerPositionPolicy, SoftmaxTrainConfig, TrainConfig,
    generate_byte_embed_softmax_with_priors, generate_byte_softmax_with_priors,
    generate_lexeme_softmax_with_memory,
    generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift,
    lexeme_quality_weights_from_vocab, run_byte_embed_softmax_training_with_model,
    run_byte_softmax_training_with_model, run_gated_mlp_backward_smoke,
    run_lexeme_embedding_training_with_model_and_quality, run_lexeme_softmax_evaluate,
    run_lexeme_softmax_training_from_softmax_model_and_quality,
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
    let mut generated_only_text = false;
    let mut mini_transformer_attention_kind = MiniTransformerAttentionKind::Base2Softmax;
    let mut mini_transformer_position_policy = MiniTransformerPositionPolicy::LearnedAbsolute;
    let mut mini_transformer_ttt_learning_rate_shift =
        nsrl_train::DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT;
    let mut mini_transformer_q_shift_explicit = false;

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
                lexeme_softmax_config.embedding_learning_rate_shift = value;
                mini_transformer_config.embedding_learning_rate_shift = value;
            }
            "--train-lexeme-embeddings" => {
                lexeme_softmax_config.train_embeddings = true;
            }
            "--lexeme-hidden-dim" => {
                lexeme_softmax_config.hidden_dim = args
                    .next()
                    .ok_or("--lexeme-hidden-dim requires an integer")?
                    .parse()?;
            }
            "--lexeme-hidden-lr-shift" => {
                lexeme_softmax_config.hidden_learning_rate_shift = args
                    .next()
                    .ok_or("--lexeme-hidden-lr-shift requires an integer")?
                    .parse()?;
            }
            "--lexeme-adapter-logit-shift" => {
                lexeme_softmax_config.adapter_logit_shift = args
                    .next()
                    .ok_or("--lexeme-adapter-logit-shift requires an integer")?
                    .parse()?;
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
            "--attention-q-lr-shift" => {
                mini_transformer_config.attention_q_learning_rate_shift = args
                    .next()
                    .ok_or("--attention-q-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_q_shift_explicit = true;
            }
            "--attention-qk-lr-shift" => {
                let value = args
                    .next()
                    .ok_or("--attention-qk-lr-shift requires an integer")?
                    .parse()?;
                mini_transformer_config.attention_qk_learning_rate_shift = value;
                if !mini_transformer_q_shift_explicit {
                    mini_transformer_config.attention_q_learning_rate_shift = value;
                }
            }
            "--adaptive-attention-shifts" => {
                mini_transformer_config.adaptive_attention_shifts = true;
                mini_transformer_config.adaptive_holographic_shifts = true;
            }
            "--adaptive-holographic-shifts" => {
                mini_transformer_config.adaptive_holographic_shifts = true;
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
            "--decode-profile" => {
                let value = args
                    .next()
                    .ok_or("--decode-profile requires coherent-prose")?;
                apply_decode_profile(&value, &mut lexeme_generation_config)?;
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
            "--mini-transformer-attention" => {
                let value = args
                    .next()
                    .ok_or(
                        "--mini-transformer-attention requires base2-softmax, linear, linear-streaming, or linear-streaming-ttt",
                    )?;
                mini_transformer_attention_kind = parse_mini_transformer_attention_kind(&value)?;
                mini_transformer_config.attention_kind = mini_transformer_attention_kind;
            }
            "--mini-transformer-position" => {
                let value = args
                    .next()
                    .ok_or("--mini-transformer-position requires learned-absolute or nope")?;
                mini_transformer_position_policy = parse_mini_transformer_position_policy(&value)?;
                mini_transformer_config.position_policy = mini_transformer_position_policy;
            }
            "--mini-transformer-ttt-lr-shift" => {
                mini_transformer_ttt_learning_rate_shift = args
                    .next()
                    .ok_or("--mini-transformer-ttt-lr-shift requires an integer")?
                    .parse()?;
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
            "--no-repeat-ngram" => {
                let value = args
                    .next()
                    .ok_or("--no-repeat-ngram requires an integer")?
                    .parse()?;
                byte_generation_config.decode.no_repeat_ngram_order = value;
                lexeme_generation_config.decode.no_repeat_ngram_order = value;
            }
            "--max-weight-delta" => {
                lexeme_softmax_config.max_weight_delta = args
                    .next()
                    .ok_or("--max-weight-delta requires an integer")?
                    .parse()?;
            }
            "--max-embedding-delta" => {
                lexeme_softmax_config.max_embedding_delta = args
                    .next()
                    .ok_or("--max-embedding-delta requires an integer")?
                    .parse()?;
            }
            "--max-hidden-weight-delta" => {
                lexeme_softmax_config.max_hidden_weight_delta = args
                    .next()
                    .ok_or("--max-hidden-weight-delta requires an integer")?
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
                    .ok_or("--quality-weight-profile requires off, cruft-aware, or prose-aware")?;
                let profile = match value.as_str() {
                    "off" => LexemeQualityWeightProfile::Off,
                    "cruft-aware" | "cruft" => LexemeQualityWeightProfile::CruftAware,
                    "prose-aware" | "prose" => LexemeQualityWeightProfile::ProseAware,
                    _ => {
                        return Err(
                            "--quality-weight-profile requires off, cruft-aware, or prose-aware"
                                .into(),
                        );
                    }
                };
                lexeme_embedding_config.quality_weight_profile = profile;
                lexeme_softmax_config.quality_weight_profile = profile;
                lexeme_generation_config.quality_weight_profile = profile;
            }
            "--lexeme-context-features" => {
                let value = args
                    .next()
                    .ok_or("--lexeme-context-features requires mean or ordered")?;
                lexeme_softmax_config.context_features = parse_lexeme_context_features(&value)?;
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
            "--corpus-prior-order" => {
                lexeme_generation_config.decode.corpus_prior_order = args
                    .next()
                    .ok_or("--corpus-prior-order requires 1, 2, or 3")?
                    .parse()?;
            }
            "--decode-frequency-cap" => {
                lexeme_generation_config.decode.frequency_penalty_cap = args
                    .next()
                    .ok_or("--decode-frequency-cap requires an integer")?
                    .parse()?;
            }
            "--decode-frequency-min-q15" => {
                lexeme_generation_config
                    .decode
                    .frequency_penalty_min_weight_q15 = args
                    .next()
                    .ok_or("--decode-frequency-min-q15 requires an integer")?
                    .parse()?;
            }
            "--decode-frequency-logit-shift" => {
                lexeme_generation_config
                    .decode
                    .frequency_penalty_logit_shift = args
                    .next()
                    .ok_or("--decode-frequency-logit-shift requires an integer")?
                    .parse()?;
            }
            "--decode-local-frequency-cap" => {
                lexeme_generation_config.decode.local_frequency_penalty_cap = args
                    .next()
                    .ok_or("--decode-local-frequency-cap requires an integer")?
                    .parse()?;
            }
            "--decode-local-frequency-min-q15" => {
                lexeme_generation_config
                    .decode
                    .local_frequency_penalty_min_weight_q15 = args
                    .next()
                    .ok_or("--decode-local-frequency-min-q15 requires an integer")?
                    .parse()?;
            }
            "--decode-local-frequency-logit-shift" => {
                lexeme_generation_config
                    .decode
                    .local_frequency_penalty_logit_shift = args
                    .next()
                    .ok_or("--decode-local-frequency-logit-shift requires an integer")?
                    .parse()?;
            }
            "--decode-local-frequency-hard-cap" => {
                lexeme_generation_config.decode.local_frequency_hard_cap = args
                    .next()
                    .ok_or("--decode-local-frequency-hard-cap requires an integer")?
                    .parse()?;
            }
            "--decode-island-count-cap" => {
                lexeme_generation_config.decode.island_penalty_count_cap = args
                    .next()
                    .ok_or("--decode-island-count-cap requires an integer")?
                    .parse()?;
            }
            "--decode-island-min-degree" => {
                lexeme_generation_config.decode.island_penalty_min_degree = args
                    .next()
                    .ok_or("--decode-island-min-degree requires an integer")?
                    .parse()?;
            }
            "--decode-island-min-q15" => {
                lexeme_generation_config
                    .decode
                    .island_penalty_min_weight_q15 = args
                    .next()
                    .ok_or("--decode-island-min-q15 requires an integer")?
                    .parse()?;
            }
            "--decode-island-logit-shift" => {
                lexeme_generation_config.decode.island_penalty_logit_shift = args
                    .next()
                    .ok_or("--decode-island-logit-shift requires an integer")?
                    .parse()?;
            }
            "--prompt-topic-radius" => {
                lexeme_generation_config.decode.prompt_topic_radius = args
                    .next()
                    .ok_or("--prompt-topic-radius requires an integer")?
                    .parse()?;
            }
            "--prompt-topic-min-q15" => {
                lexeme_generation_config.decode.prompt_topic_min_weight_q15 = args
                    .next()
                    .ok_or("--prompt-topic-min-q15 requires an integer")?
                    .parse()?;
            }
            "--prompt-topic-strict-min-q15" => {
                lexeme_generation_config
                    .decode
                    .prompt_topic_strict_min_weight_q15 = args
                    .next()
                    .ok_or("--prompt-topic-strict-min-q15 requires an integer")?
                    .parse()?;
            }
            "--prompt-topic-logit-shift" => {
                lexeme_generation_config.decode.prompt_topic_logit_shift = args
                    .next()
                    .ok_or("--prompt-topic-logit-shift requires an integer")?
                    .parse()?;
            }
            "--decode-memory-order" => {
                lexeme_generation_config.decode.memory_context_order = args
                    .next()
                    .ok_or("--decode-memory-order requires an integer")?
                    .parse()?;
            }
            "--decode-memory-min-order" => {
                lexeme_generation_config.decode.memory_min_context_order = args
                    .next()
                    .ok_or("--decode-memory-min-order requires an integer")?
                    .parse()?;
            }
            "--decode-memory-logit-shift" => {
                lexeme_generation_config.decode.memory_logit_shift = args
                    .next()
                    .ok_or("--decode-memory-logit-shift requires an integer")?
                    .parse()?;
            }
            "--strict-memory-on-steps" => {
                lexeme_generation_config.decode.strict_memory_on_steps = args
                    .next()
                    .ok_or("--strict-memory-on-steps requires an integer")?
                    .parse()?;
            }
            "--strict-memory-off-steps" => {
                lexeme_generation_config.decode.strict_memory_off_steps = args
                    .next()
                    .ok_or("--strict-memory-off-steps requires an integer")?
                    .parse()?;
            }
            "--strict-memory" => {
                lexeme_generation_config.decode.strict_memory = true;
            }
            "--strict-topic" => {
                lexeme_generation_config.decode.strict_topic = true;
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
                let value = args
                    .next()
                    .ok_or("--batch-windows requires an integer")?
                    .parse()?;
                lexeme_softmax_config.batch_windows = value;
                mini_transformer_config.batch_windows = value;
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
            "--generated-only" => {
                generated_only_text = true;
            }
            "--stop-on-sentence-terminal" => {
                lexeme_generation_config.stop_on_sentence_terminal = true;
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
                fs::write(path, run.model.try_to_bytes()?)?;
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
                generated_only_text,
            )?;
            generation.to_json_line()
        }
        "byte-embed-softmax" | "byte_embed_softmax" => {
            let path = tokens_path.ok_or("--tokens is required for byte-embed-softmax mode")?;
            let tokens = fs::read(path)?;
            let run =
                run_byte_embed_softmax_training_with_model(&tokens, byte_embed_softmax_config)?;
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
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
                generated_only_text,
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
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            run.trace.to_json_line()
        }
        "lexeme-softmax" | "lexeme_softmax" => {
            let token_path = tokens_path.ok_or("--tokens is required for lexeme-softmax mode")?;
            let model_path = model_path.ok_or("--model is required for lexeme-softmax mode")?;
            let tokens = fs::read(token_path)?;
            let model_bytes = fs::read(model_path)?;
            let vocab_size = LexemeEmbeddingModel::from_bytes(&model_bytes)
                .map(|model| model.vocab_size)
                .or_else(|_| {
                    LexemeSoftmaxModel::from_bytes(&model_bytes).map(|model| model.vocab_size)
                })?;
            let quality_weights = load_lexeme_quality_weights(
                &vocab_path,
                vocab_size,
                lexeme_softmax_config.quality_weight_profile,
            )?;
            let run = if let Ok(embedding_model) = LexemeEmbeddingModel::from_bytes(&model_bytes) {
                run_lexeme_softmax_training_with_model_and_quality(
                    &tokens,
                    embedding_model,
                    lexeme_softmax_config,
                    quality_weights.as_deref(),
                )?
            } else {
                let softmax_model = LexemeSoftmaxModel::from_bytes(&model_bytes)?;
                run_lexeme_softmax_training_from_softmax_model_and_quality(
                    &tokens,
                    softmax_model,
                    lexeme_softmax_config,
                    quality_weights.as_deref(),
                )?
            };
            if let Some(path) = model_out_path {
                fs::write(path, run.model.try_to_bytes()?)?;
            }
            run.trace.to_json_line()
        }
        "lexeme-evaluate" | "lexeme_evaluate" => {
            let token_path = tokens_path.ok_or("--tokens is required for lexeme-evaluate mode")?;
            let model_path = model_path.ok_or("--model is required for lexeme-evaluate mode")?;
            let tokens = fs::read(token_path)?;
            let model = LexemeSoftmaxModel::from_bytes(&fs::read(model_path)?)?;
            let seq_len = lexeme_softmax_config.seq_len;
            let stride = lexeme_softmax_config.stride;
            let result = run_lexeme_softmax_evaluate(
                &tokens,
                model,
                seq_len,
                stride,
                lexeme_softmax_config.max_windows,
            )?;
            result.to_json_line()
        }
        "lexeme-generate" | "lexeme_generate" => {
            let model_path = model_path.ok_or("--model is required for lexeme-generate mode")?;
            let vocab_path = vocab_path.ok_or("--vocab is required for lexeme-generate mode")?;
            if prompt.is_empty() {
                return Err("--prompt is required for lexeme-generate mode".into());
            }
            let vocab = load_lexeme_vocab(&vocab_path)?;
            let mut generation_config = lexeme_generation_config;
            configure_lexeme_sentence_stop_tokens(&mut generation_config, &vocab.entries);
            let model = LexemeSoftmaxModel::from_bytes(&fs::read(model_path)?)?;
            let prompt_tokens = encode_lexeme_prompt_tokens(&prompt, &vocab.lookup);
            let decode_priors =
                load_lexeme_decode_priors(&tokens_path, model.vocab_size, generation_config)?;
            let topic_priors = load_lexeme_topic_priors(
                &tokens_path,
                model.vocab_size,
                &prompt_tokens,
                generation_config,
            )?;
            let memory_priors = load_lexeme_memory_priors(
                &tokens_path,
                model.vocab_size,
                generation_config,
                &vocab.entries,
            )?;
            let quality_weights = load_lexeme_quality_weights(
                &Some(vocab_path),
                model.vocab_size,
                generation_config.quality_weight_profile,
            )?;
            let generation = generate_lexeme_softmax_with_memory(
                &model,
                &prompt_tokens,
                generation_config,
                decode_priors.as_ref(),
                quality_weights.as_deref(),
                topic_priors.as_ref(),
                memory_priors.as_ref(),
            )?;
            write_lexeme_text_generation(
                &text_out_path,
                &generation.prompt_tokens,
                &generation.generated_tokens,
                &vocab.entries,
                generated_only_text,
            )?;
            generation.to_json_line()
        }
        "mini-transformer-mlp" | "mini_transformer_mlp" => {
            if mini_transformer_attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
                || mini_transformer_attention_kind
                    == MiniTransformerAttentionKind::LinearStreamingTttNope
            {
                return Err(
                    "--mini-transformer-attention linear-streaming modes are generation-only; train with linear for the rescanned linear-attention backward"
                        .into(),
                );
            }
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
                fs::write(path, run.model.try_to_bytes()?)?;
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
            let generation =
                generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift(
                    &model,
                    &prompt,
                    byte_generation_config,
                    mini_transformer_attention_kind,
                    mini_transformer_position_policy,
                    decode_priors.as_ref(),
                    mini_transformer_ttt_learning_rate_shift,
                )?;
            write_text_generation(
                &text_out_path,
                &generation.prompt_bytes,
                &generation.generated_bytes,
                generated_only_text,
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
        "Usage: nsrl-train [--mode softmax|perceptron|linear-backward|gated-mlp-backward|byte-softmax|byte-generate|byte-embed-softmax|byte-embed-generate|lexeme-embedding|lexeme-softmax|lexeme-generate|mini-transformer-mlp|mini-transformer-generate] [--tokens PATH] [--model PATH] [--model-out PATH] [--vocab PATH] [--prompt TEXT] [--max-new-tokens N] [--decode greedy|sample] [--decode-profile coherent-prose|grounded-prose] [--sample-seed N] [--top-k N] [--tokenizer identity|ascii-lower] [--mini-transformer-attention base2-softmax|linear|linear-streaming|linear-streaming-ttt] [--mini-transformer-position learned-absolute|nope] [--mini-transformer-ttt-lr-shift N] [--printable-only] [--ascii-lower-only] [--repeat-window N] [--repeat-penalty-shift N] [--max-repeat-run N] [--no-repeat-ngram N] [--corpus-prior] [--corpus-prior-logit-shift N] [--corpus-prior-order 1|2|3] [--decode-frequency-cap N] [--decode-frequency-min-q15 N] [--decode-frequency-logit-shift N] [--decode-local-frequency-cap N] [--decode-local-frequency-min-q15 N] [--decode-local-frequency-logit-shift N] [--decode-local-frequency-hard-cap N] [--decode-island-count-cap N] [--decode-island-min-degree N] [--decode-island-min-q15 N] [--decode-island-logit-shift N] [--prompt-topic-radius N] [--prompt-topic-min-q15 N] [--prompt-topic-strict-min-q15 N] [--prompt-topic-logit-shift N] [--decode-memory-order N] [--decode-memory-min-order N] [--decode-memory-logit-shift N] [--strict-memory-on-steps N] [--strict-memory-off-steps N] [--strict-memory] [--strict-topic] [--strict-adjacency] [--epochs N] [--learning-rate N] [--lr-shift N] [--lr-shift-decay-windows N] [--lr-shift-decay-step N] [--max-lr-shift N] [--max-weight-delta N] [--max-embedding-delta N] [--max-hidden-weight-delta N] [--concept-frequency-cap N] [--target-frequency-cap N] [--frequency-weight-min-q15 N] [--quality-weight-profile off|cruft-aware|prose-aware] [--lexeme-context-features mean|ordered] [--train-lexeme-embeddings] [--lexeme-hidden-dim N] [--lexeme-hidden-lr-shift N] [--lexeme-adapter-logit-shift N] [--context-radius N] [--vocab-size N] [--embedding-dim N] [--mlp-lr-shift N] [--embed-lr-shift N] [--attention-lr-shift N] [--attention-q-lr-shift N] [--attention-qk-lr-shift N] [--adaptive-attention-shifts] [--adaptive-holographic-shifts] [--attention-vo-error-feedback] [--attention-vo-oracle] [--reject-loss-regression] [--seq-len N] [--stride N] [--window-offset N] [--batch-windows N] [--max-windows N] [--trace PATH] [--text-out PATH] [--generated-only] [--stop-on-sentence-terminal]"
    );
    println!();
    println!("Runs a deterministic integer training trace.");
}

fn apply_decode_profile(
    value: &str,
    config: &mut LexemeGenerationConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    match value {
        "coherent-prose" | "coherent_prose" | "prose" => {
            config.decode.strategy = DecodeStrategy::Sample;
            config.decode.top_k = 8;
            config.decode.repeat_window = 96;
            config.decode.repeat_penalty_shift = 3;
            config.decode.max_repeat_run = 2;
            config.decode.no_repeat_ngram_order = 3;
            config.decode.corpus_prior = true;
            config.decode.corpus_prior_order = 3;
            config.decode.corpus_prior_logit_shift = 7;
            config.decode.frequency_penalty_cap = 600;
            config.decode.frequency_penalty_min_weight_q15 = 6144;
            config.decode.frequency_penalty_logit_shift = 5;
            config.decode.local_frequency_penalty_cap = 2;
            config.decode.local_frequency_penalty_min_weight_q15 = 8192;
            config.decode.local_frequency_penalty_logit_shift = 4;
            config.decode.memory_context_order = 4;
            config.decode.memory_min_context_order = 2;
            config.decode.memory_logit_shift = 5;
            config.decode.strict_adjacency = true;
            config.quality_weight_profile = LexemeQualityWeightProfile::ProseAware;
            Ok(())
        }
        "grounded-prose" | "grounded_prose" | "grounded" => {
            config.decode.strategy = DecodeStrategy::Sample;
            config.decode.top_k = 8;
            config.decode.repeat_window = 96;
            config.decode.repeat_penalty_shift = 3;
            config.decode.max_repeat_run = 2;
            config.decode.no_repeat_ngram_order = 4;
            config.decode.corpus_prior = true;
            config.decode.corpus_prior_order = 3;
            config.decode.corpus_prior_logit_shift = 7;
            config.decode.frequency_penalty_cap = 600;
            config.decode.frequency_penalty_min_weight_q15 = 6144;
            config.decode.frequency_penalty_logit_shift = 5;
            config.decode.local_frequency_penalty_cap = 2;
            config.decode.local_frequency_penalty_min_weight_q15 = 8192;
            config.decode.local_frequency_penalty_logit_shift = 4;
            config.decode.memory_context_order = 4;
            config.decode.memory_min_context_order = 2;
            config.decode.memory_logit_shift = 4;
            config.decode.strict_memory_on_steps = 6;
            config.decode.strict_memory_off_steps = 3;
            config.decode.strict_memory = true;
            config.decode.strict_adjacency = true;
            config.quality_weight_profile = LexemeQualityWeightProfile::ProseAware;
            Ok(())
        }
        _ => Err("--decode-profile requires coherent-prose or grounded-prose".into()),
    }
}

fn parse_mini_transformer_attention_kind(
    value: &str,
) -> Result<MiniTransformerAttentionKind, Box<dyn std::error::Error>> {
    match value {
        "base2-softmax" | "base2_softmax" | "softmax" => {
            Ok(MiniTransformerAttentionKind::Base2Softmax)
        }
        "linear" | "linear-attention" | "linear_attention" => {
            Ok(MiniTransformerAttentionKind::Linear)
        }
        "linear-streaming"
        | "linear_streaming"
        | "linear-incremental"
        | "linear_incremental"
        | "linear-streaming-nope"
        | "linear_streaming_nope" => Ok(MiniTransformerAttentionKind::LinearStreamingNope),
        "linear-streaming-ttt"
        | "linear_streaming_ttt"
        | "linear-streaming-ttt-nope"
        | "linear_streaming_ttt_nope" => Ok(MiniTransformerAttentionKind::LinearStreamingTttNope),
        _ => Err(
            "--mini-transformer-attention requires base2-softmax, linear, linear-streaming, or linear-streaming-ttt"
                .into(),
        ),
    }
}

fn parse_mini_transformer_position_policy(
    value: &str,
) -> Result<MiniTransformerPositionPolicy, Box<dyn std::error::Error>> {
    match value {
        "learned-absolute" | "learned_absolute" | "absolute" | "position-embedding"
        | "position_embedding" => Ok(MiniTransformerPositionPolicy::LearnedAbsolute),
        "nope" | "no-position" | "no_position" | "none" => Ok(MiniTransformerPositionPolicy::Nope),
        _ => Err("--mini-transformer-position requires learned-absolute or nope".into()),
    }
}

fn parse_lexeme_context_features(
    value: &str,
) -> Result<LexemeContextFeatures, Box<dyn std::error::Error>> {
    match value {
        "mean" | "mean-context" | "mean_context" => Ok(LexemeContextFeatures::Mean),
        "ordered" | "ordered-context" | "ordered_context" => Ok(LexemeContextFeatures::Ordered),
        _ => Err("--lexeme-context-features requires mean or ordered".into()),
    }
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
        .ok_or("--vocab is required with --quality-weight-profile cruft-aware or prose-aware")?;
    let vocab = load_lexeme_vocab(path)?;
    Ok(Some(lexeme_quality_weights_from_vocab(
        &vocab.entries,
        vocab_size,
        profile,
    )?))
}

fn write_lexeme_text_generation(
    path: &Option<PathBuf>,
    prompt_tokens: &[u16],
    generated_tokens: &[u16],
    vocab: &[String],
    generated_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let capacity = if generated_only {
            generated_tokens.len()
        } else {
            prompt_tokens.len() + generated_tokens.len()
        };
        let mut all_tokens = Vec::with_capacity(capacity);
        if !generated_only {
            all_tokens.extend_from_slice(prompt_tokens);
        }
        all_tokens.extend_from_slice(generated_tokens);
        fs::write(path, render_lexeme_tokens(&all_tokens, vocab))?;
    }
    Ok(())
}

fn render_lexeme_tokens(tokens: &[u16], vocab: &[String]) -> String {
    let mut out = String::new();
    let mut previous_was_byte_word = false;
    for &token in tokens {
        if token <= u16::from(u8::MAX) {
            append_rendered_byte_token(&mut out, token as u8, previous_was_byte_word);
            previous_was_byte_word = is_render_byte_word(token as u8);
            continue;
        }

        previous_was_byte_word = false;
        let text = vocab
            .get(usize::from(token))
            .filter(|value| !value.is_empty())
            .cloned()
            .unwrap_or_else(|| String::from("?"));
        if text == LEXEME_PAGE_BOUNDARY {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }
        append_rendered_lexeme(&mut out, &text);
    }
    out
}

fn append_rendered_byte_token(out: &mut String, byte: u8, previous_was_byte_word: bool) {
    if is_render_byte_word(byte) {
        if !previous_was_byte_word
            && !out.is_empty()
            && !out.ends_with(' ')
            && !out.ends_with('\'')
            && !out.ends_with('-')
        {
            out.push(' ');
        }
        out.push(char::from(byte));
        return;
    }

    let text = String::from_utf8_lossy(&[byte]).to_string();
    append_rendered_lexeme(out, &text);
}

fn is_render_byte_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
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
    if !config.decode.corpus_prior
        && !config.decode.strict_adjacency
        && config.decode.frequency_penalty_cap == 0
        && config.decode.island_penalty_count_cap == 0
    {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with lexeme decode priors")?;
    let tokens = read_lexeme_token_file(path)?;
    Ok(Some(LexemeDecodePriors::from_tokens(&tokens, vocab_size)?))
}

fn load_lexeme_topic_priors(
    tokens_path: &Option<PathBuf>,
    vocab_size: usize,
    prompt_tokens: &[u16],
    config: LexemeGenerationConfig,
) -> Result<Option<LexemeTopicPriors>, Box<dyn std::error::Error>> {
    if config.decode.prompt_topic_radius == 0 {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with --prompt-topic-radius")?;
    let tokens = read_lexeme_token_file(path)?;
    Ok(Some(LexemeTopicPriors::from_tokens(
        &tokens,
        vocab_size,
        prompt_tokens,
        config.decode.prompt_topic_radius,
        config.decode.prompt_topic_min_weight_q15,
    )?))
}

fn load_lexeme_memory_priors(
    tokens_path: &Option<PathBuf>,
    vocab_size: usize,
    config: LexemeGenerationConfig,
    vocab: &[String],
) -> Result<Option<LexemeMemoryPriors>, Box<dyn std::error::Error>> {
    if config.decode.memory_context_order == 0 {
        return Ok(None);
    }
    let path = tokens_path
        .as_ref()
        .ok_or("--tokens is required with --decode-memory-order")?;
    let tokens = read_lexeme_token_file(path)?;
    let terminal_tokens = lexeme_terminal_tokens(vocab);
    Ok(Some(LexemeMemoryPriors::from_tokens_with_terminal_tokens(
        &tokens,
        vocab_size,
        config.decode.memory_context_order,
        &terminal_tokens,
    )?))
}

fn lexeme_terminal_tokens(vocab: &[String]) -> Vec<u16> {
    vocab
        .iter()
        .enumerate()
        .filter_map(|(index, text)| {
            if text == LEXEME_PAGE_BOUNDARY || is_sentence_terminal_lexeme(text) {
                u16::try_from(index).ok()
            } else {
                None
            }
        })
        .collect()
}

fn configure_lexeme_sentence_stop_tokens(config: &mut LexemeGenerationConfig, vocab: &[String]) {
    if !config.stop_on_sentence_terminal {
        return;
    }
    let mut tokens = [0_u16; LEXEME_SENTENCE_STOP_TOKEN_CAP];
    let mut count = 0_usize;
    for (index, text) in vocab.iter().enumerate() {
        if !is_sentence_terminal_lexeme(text) {
            continue;
        }
        let Ok(token) = u16::try_from(index) else {
            continue;
        };
        if count == LEXEME_SENTENCE_STOP_TOKEN_CAP {
            break;
        }
        tokens[count] = token;
        count += 1;
    }
    config.sentence_terminal_tokens = tokens;
    config.sentence_terminal_token_count = count;
}

fn is_sentence_terminal_lexeme(text: &str) -> bool {
    text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
}

fn read_lexeme_token_file(path: &PathBuf) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err("lexeme token stream must contain little-endian u16 tokens".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>())
}

fn write_text_generation(
    path: &Option<PathBuf>,
    prompt: &[u8],
    generated: &[u8],
    generated_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let capacity = if generated_only {
            generated.len()
        } else {
            prompt.len() + generated.len()
        };
        let mut text = Vec::with_capacity(capacity);
        if !generated_only {
            text.extend_from_slice(prompt);
        }
        text.extend_from_slice(generated);
        fs::write(path, text)?;
    }
    Ok(())
}
