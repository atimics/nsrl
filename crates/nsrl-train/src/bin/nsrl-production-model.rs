#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_core::SoftmaxNormalization;
use nsrl_corpus::subword::SubwordTokenizer;
use nsrl_train::production::{
    DirectFeatureTrainConfig, DirectHeadTrainConfig, ProductionAtomicDocumentRange,
    ProductionAtomicSourceBinding, ProductionBackwardQuantization,
    ProductionBooleanJetAnalysisRole, ProductionBooleanJetConfirmationConfig,
    ProductionBooleanJetConfirmationV2Config, ProductionBooleanJetMatchedControlV2Config,
    ProductionBooleanJetMove, ProductionBooleanJetMoveContract, ProductionBooleanJetObjectiveSpec,
    ProductionBooleanJetProtocolBindings, ProductionBooleanJetProtocolVersion,
    ProductionBooleanJetRankTwoConfig, ProductionFullTrainConfig, ProductionGenerationConfig,
    ProductionGradientAlignmentConfig, ProductionMarginOptimizerStateV1,
    ProductionMarginTrainConfig, ProductionModelConfig, ProductionModelV1,
    ProductionNumericContract, ProductionOptimizerStateV2, ProductionProjectionScales,
    ProductionSmokeConfig, ProductionTrainingNumericContract, audit_production_atomic_structure,
    audit_production_boolean_jet_confirmation, audit_production_boolean_jet_confirmation_v2,
    audit_production_boolean_jet_rank_two, audit_production_gradient_alignment,
    audit_production_probability_normalization,
    audit_production_probability_normalization_signal_attribution,
    audit_production_probability_resolution, compare_production_models, decode_bound_token_stream,
    evaluate_production_model, evaluate_production_model_canonical_nll_default_floor,
    freeze_production_atomic_structure_contract, freeze_production_boolean_jet_matched_control,
    generate_production_model, production_boolean_jet_binary_fnv64,
    production_boolean_jet_source_fnv64, train_production_direct_feature,
    train_production_direct_head_search, train_production_full_smoke,
    train_production_output_smoke, train_production_target_margin,
};

#[derive(Debug, Clone, Copy)]
struct BooleanJetMoveArg {
    group_index: usize,
    coordinate: usize,
    delta: i8,
}

type BooleanJetMoveBlocks = (
    Vec<ProductionBooleanJetMove>,
    Vec<ProductionBooleanJetMove>,
    Vec<ProductionBooleanJetMove>,
);

#[derive(Debug)]
struct Config {
    command: String,
    profile: String,
    tokenizer: Option<PathBuf>,
    tokens: Option<PathBuf>,
    source_index: Option<PathBuf>,
    model: Option<PathBuf>,
    candidate_model: Option<PathBuf>,
    model_out: Option<PathBuf>,
    generated_out: Option<PathBuf>,
    trace: Option<PathBuf>,
    optimizer_state: Option<PathBuf>,
    optimizer_state_out: Option<PathBuf>,
    seed: u64,
    output_init_amplitude: i16,
    output_forward_shift: Option<u8>,
    up_forward_shift: Option<u8>,
    context_tokens: usize,
    context_tokens_explicit: bool,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    max_new_tokens: usize,
    top_k: usize,
    stop_on_eos: bool,
    spread_windows: bool,
    targets_per_window: usize,
    training_workers: usize,
    max_windows: usize,
    epochs: usize,
    feature_shift: u8,
    bias_step_q8: i32,
    margin_q8: i32,
    matrix_learning_rate_shift: u8,
    q_learning_rate_shift: Option<u8>,
    k_learning_rate_shift: Option<u8>,
    v_learning_rate_shift: Option<u8>,
    o_learning_rate_shift: Option<u8>,
    up_learning_rate_shift: Option<u8>,
    gate_learning_rate_shift: Option<u8>,
    down_learning_rate_shift: Option<u8>,
    vector_learning_rate_shift: u8,
    output_bias_learning_rate_shift: Option<u8>,
    final_rms_learning_rate_shift: Option<u8>,
    embedding_learning_rate_shift: u8,
    embedding_learning_rate_boost_shift: u8,
    output_learning_rate_shift: u8,
    output_backward_shift: Option<u8>,
    probability_gradient_fractional_bits: u8,
    probability_normalization: SoftmaxNormalization,
    batch_windows: usize,
    max_optimizer_steps: usize,
    evaluation_windows: usize,
    reject_saturated_batch: bool,
    flush_batched_embedding_residuals: bool,
    descent_guard_windows: usize,
    descent_guard_signed_representation_blocks: bool,
    descent_guard_signed_representation_zero_saturation: bool,
    backward_quantization: ProductionBackwardQuantization,
    backward_stochastic_seed: u64,
    alignment_coordinates_per_group: usize,
    alignment_transfer_windows: usize,
    alignment_documents_per_surface: usize,
    alignment_rescue_stratified_sampling: bool,
    alignment_include_mass_corrected_no_rescue: bool,
    alignment_include_systematic_fixed_mass: bool,
    boolean_jet_expected_trunk_moves: usize,
    boolean_jet_expected_head_moves: usize,
    boolean_jet_expected_move_fingerprint: u64,
    boolean_jet_expected_manifest_hash: u64,
    boolean_jet_expected_base_model_hash: u64,
    boolean_jet_expected_tokenizer_hash: u64,
    boolean_jet_expected_token_stream_hash: u64,
    boolean_jet_expected_source_fnv64: u64,
    boolean_jet_expected_binary_fnv64: u64,
    boolean_jet_trunk_moves: Vec<BooleanJetMoveArg>,
    boolean_jet_head_moves: Vec<BooleanJetMoveArg>,
    boolean_jet_control_moves: Vec<BooleanJetMoveArg>,
    boolean_jet_proposal_document_start: usize,
    boolean_jet_proposal_documents: usize,
    boolean_jet_transfer_document_start: usize,
    boolean_jet_transfer_documents: usize,
    boolean_jet_windows_per_document: usize,
    boolean_jet_minimum_documents: usize,
    boolean_jet_reserved_document_start: usize,
}

impl Default for Config {
    fn default() -> Self {
        let smoke = ProductionSmokeConfig::default();
        let full = ProductionFullTrainConfig::default();
        Self {
            command: String::new(),
            profile: "p10m".to_string(),
            tokenizer: None,
            tokens: None,
            source_index: None,
            model: None,
            candidate_model: None,
            model_out: None,
            generated_out: None,
            trace: None,
            optimizer_state: None,
            optimizer_state_out: None,
            seed: 7,
            output_init_amplitude: 0,
            output_forward_shift: None,
            up_forward_shift: None,
            context_tokens: smoke.context_tokens,
            context_tokens_explicit: false,
            prompt: None,
            prompt_file: None,
            max_new_tokens: 64,
            top_k: 1,
            stop_on_eos: true,
            spread_windows: false,
            targets_per_window: full.targets_per_window,
            training_workers: full.training_workers,
            max_windows: smoke.max_windows,
            epochs: smoke.epochs,
            feature_shift: smoke.feature_shift,
            bias_step_q8: smoke.bias_step_q8,
            margin_q8: smoke.margin_q8,
            matrix_learning_rate_shift: full.matrix_learning_rate_shift,
            q_learning_rate_shift: full.q_learning_rate_shift,
            k_learning_rate_shift: full.k_learning_rate_shift,
            v_learning_rate_shift: full.v_learning_rate_shift,
            o_learning_rate_shift: full.o_learning_rate_shift,
            up_learning_rate_shift: full.up_learning_rate_shift,
            gate_learning_rate_shift: full.gate_learning_rate_shift,
            down_learning_rate_shift: full.down_learning_rate_shift,
            vector_learning_rate_shift: full.vector_learning_rate_shift,
            output_bias_learning_rate_shift: full.output_bias_learning_rate_shift,
            final_rms_learning_rate_shift: full.final_rms_learning_rate_shift,
            embedding_learning_rate_shift: full.embedding_learning_rate_shift,
            embedding_learning_rate_boost_shift: full.embedding_learning_rate_boost_shift,
            output_learning_rate_shift: full.output_learning_rate_shift,
            output_backward_shift: full.output_backward_shift,
            probability_gradient_fractional_bits: full.probability_gradient_fractional_bits,
            probability_normalization: full.probability_normalization,
            batch_windows: full.batch_windows,
            max_optimizer_steps: full.max_optimizer_steps,
            evaluation_windows: full.evaluation_windows,
            reject_saturated_batch: full.reject_saturated_batch,
            flush_batched_embedding_residuals: full.flush_batched_embedding_residuals,
            descent_guard_windows: full.descent_guard_windows,
            descent_guard_signed_representation_blocks: full
                .descent_guard_signed_representation_blocks,
            descent_guard_signed_representation_zero_saturation: full
                .descent_guard_signed_representation_zero_saturation,
            backward_quantization: full.backward_quantization,
            backward_stochastic_seed: full.backward_stochastic_seed,
            alignment_coordinates_per_group: 1,
            alignment_transfer_windows: 1,
            alignment_documents_per_surface: 0,
            alignment_rescue_stratified_sampling: false,
            alignment_include_mass_corrected_no_rescue: false,
            alignment_include_systematic_fixed_mass: false,
            boolean_jet_expected_trunk_moves: 4,
            boolean_jet_expected_head_moves: 2,
            boolean_jet_expected_move_fingerprint: 0,
            boolean_jet_expected_manifest_hash: 0,
            boolean_jet_expected_base_model_hash: 0,
            boolean_jet_expected_tokenizer_hash: 0,
            boolean_jet_expected_token_stream_hash: 0,
            boolean_jet_expected_source_fnv64: 0,
            boolean_jet_expected_binary_fnv64: 0,
            boolean_jet_trunk_moves: Vec::new(),
            boolean_jet_head_moves: Vec::new(),
            boolean_jet_control_moves: Vec::new(),
            boolean_jet_proposal_document_start: 8,
            boolean_jet_proposal_documents: 64,
            boolean_jet_transfer_document_start: 72,
            boolean_jet_transfer_documents: 64,
            boolean_jet_windows_per_document: 2,
            boolean_jet_minimum_documents: 32,
            boolean_jet_reserved_document_start: 136,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-production-model: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    match config.command.as_str() {
        "init" => init(config),
        "inspect" => inspect(config),
        "numeric-contract" => numeric_contract(config),
        "smoke-train" => smoke_train(config),
        "target-margin-train" => target_margin_train(config),
        "full-train-smoke" => full_train_smoke(config),
        "direct-head-train" => direct_head_train(config),
        "direct-feature-train" => direct_feature_train(config),
        "generate" => generate(config),
        "evaluate" => evaluate(config),
        "evaluate-canonical" => evaluate_canonical(config),
        "gradient-alignment-audit" => gradient_alignment_audit(config),
        "boolean-jet-rank-two-audit" => boolean_jet_rank_two_audit(config, true),
        "boolean-jet-audit" => boolean_jet_rank_two_audit(config, false),
        "boolean-jet-confirmation-audit" => boolean_jet_confirmation_audit(config),
        "boolean-jet-stability-confirmation-v2" => boolean_jet_confirmation_audit_v2(config),
        "boolean-jet-freeze-matched-control" => boolean_jet_freeze_matched_control(config),
        "boolean-jet-protocol-bindings" => boolean_jet_protocol_bindings(),
        "boolean-jet-atomic-structure-contract" => boolean_jet_atomic_structure(config, true),
        "boolean-jet-atomic-structure-audit" => boolean_jet_atomic_structure(config, false),
        "compare-evaluate" => compare_evaluate(config),
        "probability-resolution-audit" => probability_resolution_audit(config),
        "probability-normalization-audit" => probability_normalization_audit(config),
        "probability-normalization-signal-attribution-audit" => {
            probability_normalization_signal_attribution_audit(config)
        }
        "help" => {
            print_help();
            Ok(())
        }
        _ => Err(
            "expected init, inspect, numeric-contract, smoke-train, target-margin-train, full-train-smoke, direct-head-train, direct-feature-train, generate, evaluate, evaluate-canonical, gradient-alignment-audit, boolean-jet-rank-two-audit, boolean-jet-audit, boolean-jet-freeze-matched-control, boolean-jet-confirmation-audit, boolean-jet-stability-confirmation-v2, boolean-jet-protocol-bindings, boolean-jet-atomic-structure-contract, boolean-jet-atomic-structure-audit, compare-evaluate, probability-resolution-audit, probability-normalization-audit, or probability-normalization-signal-attribution-audit"
                .into(),
        ),
    }
}

fn generate(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let model_path = required(config.model, "--model")?;
    let trace_path = required(config.trace, "--trace")?;
    let prompt_bytes = match (config.prompt, config.prompt_file) {
        (Some(prompt), None) => prompt.into_bytes(),
        (None, Some(path)) => fs::read(path)?,
        (Some(_), Some(_)) => return Err("use only one of --prompt or --prompt-file".into()),
        (None, None) => return Err("generate requires --prompt or --prompt-file".into()),
    };
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let prompt_tokens = tokenizer.encode(&prompt_bytes);
    let context_tokens = if config.context_tokens_explicit {
        config.context_tokens
    } else {
        model.config.context_tokens
    };
    let trace = generate_production_model(
        &model,
        &prompt_tokens,
        ProductionGenerationConfig {
            context_tokens,
            max_new_tokens: config.max_new_tokens,
            top_k: config.top_k,
            sample_seed: config.seed,
            stop_on_eos: config.stop_on_eos,
        },
    )?;
    let generated_bytes = tokenizer.decode(&trace.generated_tokens)?;
    fs::write(trace_path, trace.to_json_line())?;
    if let Some(path) = config.generated_out {
        fs::write(path, generated_bytes)?;
    } else {
        println!("{}", String::from_utf8_lossy(&generated_bytes));
    }
    Ok(())
}

fn numeric_contract(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let (model_config, scales, contract) = if let Some(model_path) = config.model.as_ref() {
        let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
        (model.config, model.scales, model.numeric_contract()?)
    } else {
        let model_config = ProductionModelConfig::profile(&config.profile)
            .ok_or("--profile requires p10m, p20m, or p30m")?;
        let mut scales = ProductionProjectionScales::default();
        if let Some(shift) = config.output_forward_shift {
            scales.output_shift = shift;
        }
        if let Some(shift) = config.up_forward_shift {
            scales.up_shift = shift;
        }
        (
            model_config,
            scales,
            ProductionNumericContract::derive(model_config, scales)?,
        )
    };
    let training_contract = ProductionTrainingNumericContract::derive(
        model_config,
        scales,
        production_full_train_config(&config),
    )?;
    let json = format!(
        "{}{}",
        contract.to_json_line(),
        training_contract.to_json_line()
    );
    if let Some(trace_path) = config.trace.as_ref() {
        fs::write(trace_path, json)?;
    } else {
        print!("{json}");
    }
    Ok(())
}

fn probability_normalization_signal_attribution_audit(
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let source_model_path = required(config.model, "--model")?;
    let candidate_model_path = required(config.candidate_model, "--candidate-model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut source = ProductionModelV1::from_bytes(&fs::read(source_model_path)?)?;
    let mut candidate = ProductionModelV1::from_bytes(&fs::read(candidate_model_path)?)?;
    if tokenizer.tokenizer_hash() != source.tokenizer_hash
        || tokenizer.tokenizer_hash() != candidate.tokenizer_hash
        || tokenizer.vocab_size() != source.config.vocab_size
        || tokenizer.vocab_size() != candidate.config.vocab_size
    {
        return Err("models and tokenizer binding mismatch".into());
    }
    if let Some(shift) = config.up_forward_shift {
        source.scales.up_shift = shift;
        candidate.scales.up_shift = shift;
        source.validate()?;
        candidate.validate()?;
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let trace = audit_production_probability_normalization_signal_attribution(
        &source,
        &candidate,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn probability_normalization_audit(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let source_model_path = required(config.model, "--model")?;
    let candidate_model_path = required(config.candidate_model, "--candidate-model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut source = ProductionModelV1::from_bytes(&fs::read(source_model_path)?)?;
    let mut candidate = ProductionModelV1::from_bytes(&fs::read(candidate_model_path)?)?;
    if tokenizer.tokenizer_hash() != source.tokenizer_hash
        || tokenizer.tokenizer_hash() != candidate.tokenizer_hash
        || tokenizer.vocab_size() != source.config.vocab_size
        || tokenizer.vocab_size() != candidate.config.vocab_size
    {
        return Err("models and tokenizer binding mismatch".into());
    }
    if let Some(shift) = config.up_forward_shift {
        source.scales.up_shift = shift;
        candidate.scales.up_shift = shift;
        source.validate()?;
        candidate.validate()?;
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let trace = audit_production_probability_normalization(
        &source,
        &candidate,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn probability_resolution_audit(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let source_model_path = required(config.model, "--model")?;
    let candidate_model_path = required(config.candidate_model, "--candidate-model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut source = ProductionModelV1::from_bytes(&fs::read(source_model_path)?)?;
    let mut candidate = ProductionModelV1::from_bytes(&fs::read(candidate_model_path)?)?;
    if tokenizer.tokenizer_hash() != source.tokenizer_hash
        || tokenizer.tokenizer_hash() != candidate.tokenizer_hash
        || tokenizer.vocab_size() != source.config.vocab_size
        || tokenizer.vocab_size() != candidate.config.vocab_size
    {
        return Err("models and tokenizer binding mismatch".into());
    }
    if let Some(shift) = config.up_forward_shift {
        source.scales.up_shift = shift;
        candidate.scales.up_shift = shift;
        source.validate()?;
        candidate.validate()?;
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let trace = audit_production_probability_resolution(
        &source,
        &candidate,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn compare_evaluate(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let source_model_path = required(config.model, "--model")?;
    let candidate_model_path = required(config.candidate_model, "--candidate-model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut source = ProductionModelV1::from_bytes(&fs::read(source_model_path)?)?;
    let mut candidate = ProductionModelV1::from_bytes(&fs::read(candidate_model_path)?)?;
    if tokenizer.tokenizer_hash() != source.tokenizer_hash
        || tokenizer.tokenizer_hash() != candidate.tokenizer_hash
        || tokenizer.vocab_size() != source.config.vocab_size
        || tokenizer.vocab_size() != candidate.config.vocab_size
    {
        return Err("models and tokenizer binding mismatch".into());
    }
    if let Some(shift) = config.up_forward_shift {
        source.scales.up_shift = shift;
        candidate.scales.up_shift = shift;
        source.validate()?;
        candidate.validate()?;
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        source.tokenizer_hash,
        source.config.vocab_size,
    )?;
    let trace = compare_production_models(
        &source,
        &candidate,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn evaluate(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = evaluate_production_model(
        &model,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn evaluate_canonical(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = evaluate_production_model_canonical_nll_default_floor(
        &model,
        &tokens,
        token_stream_hash,
        config.context_tokens,
        config.max_windows,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn gradient_alignment_audit(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let training_config = production_full_train_config(&config);
    let alignment_config = production_gradient_alignment_config(&config);
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = audit_production_gradient_alignment(
        &model,
        &tokens,
        token_stream_hash,
        training_config,
        alignment_config,
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn boolean_jet_rank_two_audit(
    config: Config,
    legacy_schema: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let training_config = production_full_train_config(&config);
    let jet_config = ProductionBooleanJetRankTwoConfig {
        alignment: production_gradient_alignment_config(&config),
        expected_trunk_moves: config.boolean_jet_expected_trunk_moves,
        expected_head_moves: config.boolean_jet_expected_head_moves,
        expected_move_fingerprint: config.boolean_jet_expected_move_fingerprint,
    };
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = audit_production_boolean_jet_rank_two(
        &model,
        &tokens,
        token_stream_hash,
        training_config,
        jet_config,
    )?;
    fs::write(
        trace_path,
        if legacy_schema {
            trace.to_legacy_json_line()
        } else {
            trace.to_json_line()
        },
    )?;
    Ok(())
}

fn boolean_jet_confirmation_audit(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer.clone(), "--tokenizer")?;
    let tokens_path = required(config.tokens.clone(), "--tokens")?;
    let model_path = required(config.model.clone(), "--model")?;
    let trace_path = required(config.trace.clone(), "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let (trunk_moves, head_moves, matched_control_moves) = explicit_boolean_jet_moves(&config)?;
    let trace = audit_production_boolean_jet_confirmation(
        &model,
        &tokens,
        token_stream_hash,
        ProductionBooleanJetConfirmationConfig {
            context_tokens: config.context_tokens,
            objective: ProductionBooleanJetObjectiveSpec::wide_q47_v1(),
            move_contract: ProductionBooleanJetMoveContract {
                protocol_version: ProductionBooleanJetProtocolVersion::ConfirmationV1,
                analysis_role: ProductionBooleanJetAnalysisRole::Confirmation,
                expected_source_fnv64: 0,
                expected_binary_fnv64: 0,
                expected_base_model_hash: config.boolean_jet_expected_base_model_hash,
                expected_tokenizer_hash: config.boolean_jet_expected_tokenizer_hash,
                expected_token_stream_hash: config.boolean_jet_expected_token_stream_hash,
                expected_move_fingerprint: config.boolean_jet_expected_move_fingerprint,
                expected_manifest_hash: config.boolean_jet_expected_manifest_hash,
                trunk_moves,
                head_moves,
                matched_control_moves,
            },
            proposal_document_start: config.boolean_jet_proposal_document_start,
            proposal_documents: config.boolean_jet_proposal_documents,
            transfer_document_start: config.boolean_jet_transfer_document_start,
            transfer_documents: config.boolean_jet_transfer_documents,
            windows_per_document: config.boolean_jet_windows_per_document,
            minimum_independent_documents: config.boolean_jet_minimum_documents,
            significance_numerator: 1,
            significance_denominator: 20,
        },
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn boolean_jet_confirmation_audit_v2(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer.clone(), "--tokenizer")?;
    let tokens_path = required(config.tokens.clone(), "--tokens")?;
    let model_path = required(config.model.clone(), "--model")?;
    let trace_path = required(config.trace.clone(), "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let protocol_bindings = current_boolean_jet_protocol_bindings()?;
    if config.boolean_jet_expected_source_fnv64 != protocol_bindings.source_fnv64
        || config.boolean_jet_expected_binary_fnv64 != protocol_bindings.binary_fnv64
    {
        return Err("source/binary protocol binding mismatch".into());
    }
    let (trunk_moves, head_moves, matched_control_moves) = explicit_boolean_jet_moves(&config)?;
    let trace = audit_production_boolean_jet_confirmation_v2(
        &model,
        &tokens,
        token_stream_hash,
        ProductionBooleanJetConfirmationV2Config {
            primary: ProductionBooleanJetConfirmationConfig {
                context_tokens: config.context_tokens,
                objective: ProductionBooleanJetObjectiveSpec::wide_q47_q32_v2(),
                move_contract: ProductionBooleanJetMoveContract {
                    protocol_version: ProductionBooleanJetProtocolVersion::StabilityV2,
                    analysis_role: ProductionBooleanJetAnalysisRole::Confirmation,
                    expected_source_fnv64: config.boolean_jet_expected_source_fnv64,
                    expected_binary_fnv64: config.boolean_jet_expected_binary_fnv64,
                    expected_base_model_hash: config.boolean_jet_expected_base_model_hash,
                    expected_tokenizer_hash: config.boolean_jet_expected_tokenizer_hash,
                    expected_token_stream_hash: config.boolean_jet_expected_token_stream_hash,
                    expected_move_fingerprint: config.boolean_jet_expected_move_fingerprint,
                    expected_manifest_hash: config.boolean_jet_expected_manifest_hash,
                    trunk_moves,
                    head_moves,
                    matched_control_moves,
                },
                proposal_document_start: config.boolean_jet_proposal_document_start,
                proposal_documents: config.boolean_jet_proposal_documents,
                transfer_document_start: config.boolean_jet_transfer_document_start,
                transfer_documents: config.boolean_jet_transfer_documents,
                windows_per_document: config.boolean_jet_windows_per_document,
                minimum_independent_documents: config.boolean_jet_minimum_documents,
                significance_numerator: 1,
                significance_denominator: 20,
            },
            robustness_objective: ProductionBooleanJetObjectiveSpec::canonical_q15_v1(),
            protocol_bindings,
            reserved_document_start: config.boolean_jet_reserved_document_start,
        },
    )?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn boolean_jet_protocol_bindings() -> Result<(), Box<dyn std::error::Error>> {
    let bindings = current_boolean_jet_protocol_bindings()?;
    println!(
        "{{\"schema\":\"nsrl.production_boolean_jet_protocol_bindings.v2\",\"source_fnv64\":\"0x{:016x}\",\"binary_fnv64\":\"0x{:016x}\"}}",
        bindings.source_fnv64, bindings.binary_fnv64,
    );
    Ok(())
}

fn current_boolean_jet_protocol_bindings()
-> Result<ProductionBooleanJetProtocolBindings, Box<dyn std::error::Error>> {
    let executable = env::current_exe()?;
    Ok(ProductionBooleanJetProtocolBindings {
        source_fnv64: production_boolean_jet_source_fnv64(),
        binary_fnv64: production_boolean_jet_binary_fnv64(&fs::read(executable)?),
    })
}

fn boolean_jet_atomic_structure(
    config: Config,
    freeze_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer.clone(), "--tokenizer")?;
    let tokens_path = required(config.tokens.clone(), "--tokens")?;
    let source_index_path = required(config.source_index.clone(), "--source-index")?;
    let model_path = required(config.model.clone(), "--model")?;
    let trace_path = required(config.trace.clone(), "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let source_binding = atomic_source_binding(
        &fs::read(source_index_path)?,
        config.boolean_jet_proposal_document_start,
        config.boolean_jet_proposal_documents,
    )?;
    let protocol_bindings = current_boolean_jet_protocol_bindings()?;
    if config.boolean_jet_expected_base_model_hash != model.model_hash()
        || config.boolean_jet_expected_tokenizer_hash != model.tokenizer_hash
        || config.boolean_jet_expected_token_stream_hash != token_stream_hash
        || config.boolean_jet_expected_source_fnv64 != protocol_bindings.source_fnv64
        || config.boolean_jet_expected_binary_fnv64 != protocol_bindings.binary_fnv64
    {
        return Err("atomic-structure artifact/protocol binding mismatch".into());
    }
    let (trunk_moves, head_moves, controls) = explicit_boolean_jet_moves(&config)?;
    if !controls.is_empty()
        || trunk_moves.len() != 4
        || head_moves.len() != 2
        || config.boolean_jet_windows_per_document != 2
    {
        return Err("atomic-structure audit requires exactly four trunk and two head atoms".into());
    }
    let moves = trunk_moves
        .into_iter()
        .chain(head_moves)
        .collect::<Vec<_>>();
    let contract = freeze_production_atomic_structure_contract(
        &model,
        token_stream_hash,
        config.context_tokens,
        moves,
        protocol_bindings,
        source_binding,
        ProductionAtomicDocumentRange {
            start: config.boolean_jet_proposal_document_start,
            count: config.boolean_jet_proposal_documents,
        },
    )?;
    if config.boolean_jet_expected_move_fingerprint != 0
        && config.boolean_jet_expected_move_fingerprint != contract.move_fingerprint
    {
        return Err("atomic-structure move fingerprint mismatch".into());
    }
    if freeze_only {
        fs::write(trace_path, contract.to_json_line())?;
        return Ok(());
    }
    if config.boolean_jet_expected_manifest_hash == 0
        || config.boolean_jet_expected_manifest_hash != contract.manifest_hash
    {
        return Err("atomic-structure frozen manifest mismatch".into());
    }
    let trace = audit_production_atomic_structure(&model, &tokens, token_stream_hash, contract)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn atomic_source_binding(
    source_index: &[u8],
    document_start: usize,
    documents: usize,
) -> Result<ProductionAtomicSourceBinding, Box<dyn std::error::Error>> {
    let index = std::str::from_utf8(source_index)?;
    let mut lines = index.lines();
    if lines.next() != Some("schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256") {
        return Err("atomic-structure source index has an unsupported header".into());
    }
    let rows = lines.collect::<Vec<_>>();
    if rows.len() < document_start + documents {
        return Err("atomic-structure source index does not cover the requested documents".into());
    }
    let mut clusters = BTreeSet::new();
    for row in &rows[document_start..document_start + documents] {
        let fields = row.split('\t').collect::<Vec<_>>();
        if fields.len() != 7
            || fields[0] != "nsrl.production_corpus_record.v1"
            || fields[1] != "dev"
        {
            return Err("atomic-structure source index contains an invalid proposal row".into());
        }
        let mut identity = fields[2].rsplitn(3, ':');
        let _document_hash = identity.next();
        let _source_document = identity.next();
        let source_cluster = identity
            .next()
            .ok_or("atomic-structure document identity has no source cluster")?;
        if source_cluster.is_empty() {
            return Err("atomic-structure document identity has an empty source cluster".into());
        }
        clusters.insert(source_cluster);
    }
    let mut canonical_clusters = b"nsrl.production_atomic_source_clusters.v1\0".to_vec();
    for cluster in &clusters {
        canonical_clusters.extend_from_slice(&(cluster.len() as u64).to_le_bytes());
        canonical_clusters.extend_from_slice(cluster.as_bytes());
    }
    Ok(ProductionAtomicSourceBinding {
        source_index_hash: production_boolean_jet_binary_fnv64(source_index),
        proposal_source_cluster_hash: production_boolean_jet_binary_fnv64(&canonical_clusters),
        proposal_source_clusters: clusters.len(),
    })
}

fn boolean_jet_freeze_matched_control(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer.clone(), "--tokenizer")?;
    let tokens_path = required(config.tokens.clone(), "--tokens")?;
    let model_path = required(config.model.clone(), "--model")?;
    let trace_path = required(config.trace.clone(), "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
        || config.boolean_jet_expected_base_model_hash != model.model_hash()
        || config.boolean_jet_expected_tokenizer_hash != model.tokenizer_hash
    {
        return Err("frozen model/tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    if config.boolean_jet_expected_token_stream_hash != token_stream_hash {
        return Err("frozen token-stream binding mismatch".into());
    }
    let protocol_bindings = current_boolean_jet_protocol_bindings()?;
    if config.boolean_jet_expected_source_fnv64 != protocol_bindings.source_fnv64
        || config.boolean_jet_expected_binary_fnv64 != protocol_bindings.binary_fnv64
    {
        return Err("frozen source/binary binding mismatch".into());
    }
    let (trunk_moves, head_moves, _) = explicit_boolean_jet_moves(&config)?;
    let manifest = freeze_production_boolean_jet_matched_control(
        &model,
        &tokens,
        token_stream_hash,
        &trunk_moves,
        &head_moves,
        ProductionBooleanJetMatchedControlV2Config {
            context_tokens: config.context_tokens,
            visibility_document_start: config.boolean_jet_proposal_document_start,
            visibility_documents: config.boolean_jet_proposal_documents,
            windows_per_document: config.boolean_jet_windows_per_document,
            reserved_document_start: config.boolean_jet_reserved_document_start,
            seed: config.seed,
            protocol_bindings,
        },
    )?;
    if config.boolean_jet_expected_move_fingerprint != 0
        && manifest.move_fingerprint != config.boolean_jet_expected_move_fingerprint
    {
        return Err("frozen move fingerprint mismatch".into());
    }
    fs::write(trace_path, manifest.to_json_line())?;
    Ok(())
}

fn explicit_boolean_jet_moves(
    config: &Config,
) -> Result<BooleanJetMoveBlocks, Box<dyn std::error::Error>> {
    const GROUPS: [&str; 13] = [
        "embeddings",
        "attention_rms",
        "mlp_rms",
        "final_rms",
        "q",
        "k",
        "v",
        "o",
        "up",
        "gate",
        "down",
        "output",
        "bias",
    ];
    if config.boolean_jet_trunk_moves.is_empty() || config.boolean_jet_head_moves.is_empty() {
        return Err(
            "Boolean-jet commands require explicit --trunk-move and --head-move atoms".into(),
        );
    }
    let mut trunk = config.boolean_jet_trunk_moves.clone();
    let mut head = config.boolean_jet_head_moves.clone();
    let mut control = config.boolean_jet_control_moves.clone();
    trunk.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    head.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    control.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    let trunk_len = trunk.len();
    let build = |block: &'static str,
                 order_offset: usize,
                 atoms: Vec<BooleanJetMoveArg>|
     -> Result<Vec<ProductionBooleanJetMove>, Box<dyn std::error::Error>> {
        atoms
            .into_iter()
            .enumerate()
            .map(|(index, atom)| {
                let group = GROUPS
                    .get(atom.group_index)
                    .copied()
                    .ok_or("Boolean-jet move group index exceeds production groups")?;
                Ok(ProductionBooleanJetMove {
                    block,
                    group,
                    group_index: atom.group_index,
                    coordinate: atom.coordinate,
                    parameter_delta: atom.delta,
                    coarse_gradient: 0,
                    selection_strata: vec!["frozen_v3_manifest"],
                    source_lane: "mass-corrected-normalized-rhu",
                    move_kind: "model_only_unit_sign_probe",
                    canonical_order: order_offset + index,
                })
            })
            .collect()
    };
    let head_len = head.len();
    let trunk_moves = build("trunk", 0, trunk)?;
    let head_moves = build("head", trunk_len, head)?;
    let mut control_moves = build("matched_control", trunk_len + head_len, control)?;
    for movement in &mut control_moves {
        movement.source_lane = "seeded-matched-random-control-v2";
        movement.selection_strata =
            vec!["frozen_group_cardinality_width_visibility_margin_control"];
    }
    Ok((trunk_moves, head_moves, control_moves))
}

fn production_gradient_alignment_config(config: &Config) -> ProductionGradientAlignmentConfig {
    ProductionGradientAlignmentConfig {
        proposal_windows: config.max_windows,
        transfer_windows: config.alignment_transfer_windows,
        documents_per_surface: config.alignment_documents_per_surface,
        rescue_stratified_sampling: config.alignment_rescue_stratified_sampling,
        include_mass_corrected_no_rescue: config.alignment_include_mass_corrected_no_rescue,
        include_systematic_fixed_mass: config.alignment_include_systematic_fixed_mass,
        coordinates_per_group: config.alignment_coordinates_per_group,
        sample_seed: config.seed,
    }
}

fn direct_head_train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let head_config = DirectHeadTrainConfig {
        context_tokens: config.context_tokens,
        train_windows: config.max_windows,
        dev_windows: config.evaluation_windows,
        candidates_per_round: config.alignment_coordinates_per_group,
        max_rounds: config.max_optimizer_steps,
        min_train_nll_delta: 0,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization,
        sample_seed: config.seed,
    };
    let trace =
        train_production_direct_head_search(&mut model, &tokens, token_stream_hash, head_config)?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn direct_feature_train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let feature_config = DirectFeatureTrainConfig {
        context_tokens: config.context_tokens,
        train_windows: config.max_windows,
        dev_windows: config.evaluation_windows,
        head_candidates_per_round: config.alignment_coordinates_per_group,
        final_rms_candidates_per_round: config.alignment_coordinates_per_group.clamp(8, 64),
        max_rounds: config.max_optimizer_steps,
        min_train_nll_delta: 0,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization,
        sample_seed: config.seed,
    };
    let trace =
        train_production_direct_feature(&mut model, &tokens, token_stream_hash, feature_config)?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn full_train_smoke(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let training_config = production_full_train_config(&config);
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let optimizer_out = required(config.optimizer_state_out, "--optimizer-state-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let optimizer = config
        .optimizer_state
        .map(fs::read)
        .transpose()?
        .map(|bytes| ProductionOptimizerStateV2::from_bytes(&bytes))
        .transpose()?;
    let (trace, optimizer) = train_production_full_smoke(
        &mut model,
        &tokens,
        token_stream_hash,
        training_config,
        optimizer,
    )?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(optimizer_out, optimizer.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn target_margin_train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let training_config = ProductionMarginTrainConfig {
        context_tokens: config.context_tokens,
        max_windows: config.max_windows,
        spread_windows: config.spread_windows,
        targets_per_window: config.targets_per_window,
        training_workers: config.training_workers,
        epochs: config.epochs,
        feature_shift: config.feature_shift,
        margin_q8: config.margin_q8,
        batch_windows: config.batch_windows,
        max_optimizer_steps: config.max_optimizer_steps,
        evaluation_windows: config.evaluation_windows,
    };
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let optimizer_out = required(config.optimizer_state_out, "--optimizer-state-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let optimizer = config
        .optimizer_state
        .map(fs::read)
        .transpose()?
        .map(|bytes| ProductionMarginOptimizerStateV1::from_bytes(&bytes))
        .transpose()?;
    let (trace, optimizer) = train_production_target_margin(
        &mut model,
        &tokens,
        token_stream_hash,
        training_config,
        optimizer,
    )?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(optimizer_out, optimizer.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn production_full_train_config(config: &Config) -> ProductionFullTrainConfig {
    ProductionFullTrainConfig {
        context_tokens: config.context_tokens,
        max_windows: config.max_windows,
        spread_windows: config.spread_windows,
        targets_per_window: config.targets_per_window,
        training_workers: config.training_workers,
        epochs: config.epochs,
        matrix_learning_rate_shift: config.matrix_learning_rate_shift,
        q_learning_rate_shift: config.q_learning_rate_shift,
        k_learning_rate_shift: config.k_learning_rate_shift,
        v_learning_rate_shift: config.v_learning_rate_shift,
        o_learning_rate_shift: config.o_learning_rate_shift,
        up_learning_rate_shift: config.up_learning_rate_shift,
        gate_learning_rate_shift: config.gate_learning_rate_shift,
        down_learning_rate_shift: config.down_learning_rate_shift,
        vector_learning_rate_shift: config.vector_learning_rate_shift,
        output_bias_learning_rate_shift: config.output_bias_learning_rate_shift,
        final_rms_learning_rate_shift: config.final_rms_learning_rate_shift,
        embedding_learning_rate_shift: config.embedding_learning_rate_shift,
        embedding_learning_rate_boost_shift: config.embedding_learning_rate_boost_shift,
        output_learning_rate_shift: config.output_learning_rate_shift,
        output_backward_shift: config.output_backward_shift,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization,
        batch_windows: config.batch_windows,
        max_optimizer_steps: config.max_optimizer_steps,
        evaluation_windows: config.evaluation_windows,
        reject_saturated_batch: config.reject_saturated_batch,
        flush_batched_embedding_residuals: config.flush_batched_embedding_residuals,
        descent_guard_windows: config.descent_guard_windows,
        descent_guard_signed_representation_blocks: config
            .descent_guard_signed_representation_blocks,
        descent_guard_signed_representation_zero_saturation: config
            .descent_guard_signed_representation_zero_saturation,
        backward_quantization: config.backward_quantization,
        backward_stochastic_seed: config.backward_stochastic_seed,
    }
}

fn init(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let model_config = ProductionModelConfig::profile(&config.profile)
        .ok_or("--profile requires p10m, p20m, or p30m")?;
    if tokenizer.vocab_size() != model_config.vocab_size {
        return Err("tokenizer vocabulary does not match production profile".into());
    }
    let mut model =
        ProductionModelV1::new_initial(model_config, tokenizer.tokenizer_hash(), config.seed)?;
    model.initialize_output_weights(config.output_init_amplitude)?;
    if let Some(output_forward_shift) = config.output_forward_shift {
        model.scales.output_shift = output_forward_shift;
        model.validate()?;
    }
    if let Some(up_forward_shift) = config.up_forward_shift {
        model.scales.up_shift = up_forward_shift;
        model.validate()?;
    }
    let bytes = model.try_to_bytes()?;
    fs::write(model_out, &bytes)?;
    fs::write(
        trace_path,
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_init.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"tokenizer_hash\":\"0x{:016x}\",\"initialization_seed\":{},",
                "\"model_hash\":\"0x{:016x}\",\"artifact_bytes\":{},\"output_init_amplitude\":{},",
                "\"output_forward_shift\":{},\"up_forward_shift\":{},",
                "\"full_layer_backward_ready\":true,\"output_head_smoke_ready\":true}}\n"
            ),
            model.config.profile_id().unwrap_or("custom"),
            model.parameter_count(),
            model.tokenizer_hash,
            model.initialization_seed,
            model.model_hash(),
            bytes.len(),
            config.output_init_amplitude,
            model.scales.output_shift,
            model.scales.up_shift,
        ),
    )?;
    Ok(())
}

fn inspect(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = required(config.model, "--model")?;
    let bytes = fs::read(model_path)?;
    let model = ProductionModelV1::from_bytes(&bytes)?;
    println!(
        concat!(
            "{{\"schema\":\"nsrl.production_model_inspect.v1\",",
            "\"profile\":\"{}\",\"vocab_size\":{},\"d_model\":{},\"heads\":{},",
            "\"layers\":{},\"hidden_dim\":{},\"context_tokens\":{},",
            "\"parameter_count\":{},\"tokenizer_hash\":\"0x{:016x}\",",
            "\"initialization_seed\":{},\"model_hash\":\"0x{:016x}\",\"artifact_bytes\":{},\"output_forward_shift\":{}}}"
        ),
        model.config.profile_id().unwrap_or("custom"),
        model.config.vocab_size,
        model.config.d_model,
        model.config.heads,
        model.config.layers,
        model.config.hidden_dim,
        model.config.context_tokens,
        model.parameter_count(),
        model.tokenizer_hash,
        model.initialization_seed,
        model.model_hash(),
        bytes.len(),
        model.scales.output_shift,
    );
    Ok(())
}

fn smoke_train(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = required(config.tokenizer, "--tokenizer")?;
    let tokens_path = required(config.tokens, "--tokens")?;
    let model_path = required(config.model, "--model")?;
    let model_out = required(config.model_out, "--model-out")?;
    let trace_path = required(config.trace, "--trace")?;
    let tokenizer = SubwordTokenizer::from_bytes(&fs::read(tokenizer_path)?)?;
    let mut model = ProductionModelV1::from_bytes(&fs::read(model_path)?)?;
    if tokenizer.tokenizer_hash() != model.tokenizer_hash
        || tokenizer.vocab_size() != model.config.vocab_size
    {
        return Err("model and tokenizer binding mismatch".into());
    }
    let (tokens, token_stream_hash) = decode_bound_token_stream(
        &fs::read(tokens_path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    let trace = train_production_output_smoke(
        &mut model,
        &tokens,
        token_stream_hash,
        ProductionSmokeConfig {
            context_tokens: config.context_tokens,
            max_windows: config.max_windows,
            epochs: config.epochs,
            feature_shift: config.feature_shift,
            bias_step_q8: config.bias_step_q8,
            margin_q8: config.margin_q8,
            spread_windows: config.spread_windows,
        },
    )?;
    fs::write(model_out, model.try_to_bytes()?)?;
    fs::write(trace_path, trace.to_json_line())?;
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "init"
            | "help"
            | "inspect"
            | "numeric-contract"
            | "smoke-train"
            | "target-margin-train"
            | "full-train-smoke"
            | "direct-head-train"
            | "direct-feature-train"
            | "generate"
            | "evaluate"
            | "evaluate-canonical"
            | "gradient-alignment-audit"
            | "boolean-jet-rank-two-audit"
            | "boolean-jet-audit"
            | "boolean-jet-freeze-matched-control"
            | "boolean-jet-confirmation-audit"
            | "boolean-jet-stability-confirmation-v2"
            | "boolean-jet-protocol-bindings"
            | "boolean-jet-atomic-structure-contract"
            | "boolean-jet-atomic-structure-audit"
            | "compare-evaluate"
            | "probability-resolution-audit"
            | "probability-normalization-audit"
            | "probability-normalization-signal-attribution-audit"
                if config.command.is_empty() =>
            {
                config.command = arg
            }
            "--profile" => config.profile = next(&mut args, "--profile")?,
            "--tokenizer" => {
                config.tokenizer = Some(PathBuf::from(next(&mut args, "--tokenizer")?))
            }
            "--tokens" => config.tokens = Some(PathBuf::from(next(&mut args, "--tokens")?)),
            "--source-index" => {
                config.source_index = Some(PathBuf::from(next(&mut args, "--source-index")?))
            }
            "--model" => config.model = Some(PathBuf::from(next(&mut args, "--model")?)),
            "--candidate-model" => {
                config.candidate_model = Some(PathBuf::from(next(&mut args, "--candidate-model")?))
            }
            "--model-out" => {
                config.model_out = Some(PathBuf::from(next(&mut args, "--model-out")?))
            }
            "--generated-out" => {
                config.generated_out = Some(PathBuf::from(next(&mut args, "--generated-out")?))
            }
            "--trace" => config.trace = Some(PathBuf::from(next(&mut args, "--trace")?)),
            "--optimizer-state" => {
                config.optimizer_state = Some(PathBuf::from(next(&mut args, "--optimizer-state")?))
            }
            "--optimizer-state-out" => {
                config.optimizer_state_out =
                    Some(PathBuf::from(next(&mut args, "--optimizer-state-out")?))
            }
            "--seed" => config.seed = next(&mut args, "--seed")?.parse()?,
            "--output-init-amplitude" => {
                config.output_init_amplitude =
                    next(&mut args, "--output-init-amplitude")?.parse()?
            }
            "--output-forward-shift" => {
                config.output_forward_shift =
                    Some(next(&mut args, "--output-forward-shift")?.parse()?)
            }
            "--up-forward-shift" => {
                config.up_forward_shift = Some(next(&mut args, "--up-forward-shift")?.parse()?)
            }
            "--context-tokens" => {
                config.context_tokens = next(&mut args, "--context-tokens")?.parse()?;
                config.context_tokens_explicit = true;
            }
            "--prompt" => config.prompt = Some(next(&mut args, "--prompt")?),
            "--prompt-file" => {
                config.prompt_file = Some(PathBuf::from(next(&mut args, "--prompt-file")?))
            }
            "--max-new-tokens" => {
                config.max_new_tokens = next(&mut args, "--max-new-tokens")?.parse()?
            }
            "--top-k" => config.top_k = next(&mut args, "--top-k")?.parse()?,
            "--no-stop-on-eos" => config.stop_on_eos = false,
            "--spread-windows" => config.spread_windows = true,
            "--targets-per-window" => {
                config.targets_per_window = next(&mut args, "--targets-per-window")?.parse()?
            }
            "--training-workers" => {
                config.training_workers = next(&mut args, "--training-workers")?.parse()?
            }
            "--max-windows" => config.max_windows = next(&mut args, "--max-windows")?.parse()?,
            "--epochs" => config.epochs = next(&mut args, "--epochs")?.parse()?,
            "--feature-shift" => {
                config.feature_shift = next(&mut args, "--feature-shift")?.parse()?
            }
            "--bias-step-q8" => config.bias_step_q8 = next(&mut args, "--bias-step-q8")?.parse()?,
            "--margin-q8" => config.margin_q8 = next(&mut args, "--margin-q8")?.parse()?,
            "--matrix-learning-rate-shift" => {
                config.matrix_learning_rate_shift =
                    next(&mut args, "--matrix-learning-rate-shift")?.parse()?
            }
            "--q-learning-rate-shift" => {
                config.q_learning_rate_shift =
                    Some(next(&mut args, "--q-learning-rate-shift")?.parse()?)
            }
            "--k-learning-rate-shift" => {
                config.k_learning_rate_shift =
                    Some(next(&mut args, "--k-learning-rate-shift")?.parse()?)
            }
            "--v-learning-rate-shift" => {
                config.v_learning_rate_shift =
                    Some(next(&mut args, "--v-learning-rate-shift")?.parse()?)
            }
            "--o-learning-rate-shift" => {
                config.o_learning_rate_shift =
                    Some(next(&mut args, "--o-learning-rate-shift")?.parse()?)
            }
            "--up-learning-rate-shift" => {
                config.up_learning_rate_shift =
                    Some(next(&mut args, "--up-learning-rate-shift")?.parse()?)
            }
            "--gate-learning-rate-shift" => {
                config.gate_learning_rate_shift =
                    Some(next(&mut args, "--gate-learning-rate-shift")?.parse()?)
            }
            "--down-learning-rate-shift" => {
                config.down_learning_rate_shift =
                    Some(next(&mut args, "--down-learning-rate-shift")?.parse()?)
            }
            "--vector-learning-rate-shift" => {
                config.vector_learning_rate_shift =
                    next(&mut args, "--vector-learning-rate-shift")?.parse()?
            }
            "--output-bias-learning-rate-shift" => {
                config.output_bias_learning_rate_shift =
                    Some(next(&mut args, "--output-bias-learning-rate-shift")?.parse()?)
            }
            "--embedding-learning-rate-shift" => {
                config.embedding_learning_rate_shift =
                    next(&mut args, "--embedding-learning-rate-shift")?.parse()?
            }
            "--embedding-learning-rate-boost-shift" => {
                config.embedding_learning_rate_boost_shift =
                    next(&mut args, "--embedding-learning-rate-boost-shift")?.parse()?
            }
            "--output-learning-rate-shift" => {
                config.output_learning_rate_shift =
                    next(&mut args, "--output-learning-rate-shift")?.parse()?
            }
            "--final-rms-learning-rate-shift" => {
                config.final_rms_learning_rate_shift =
                    Some(next(&mut args, "--final-rms-learning-rate-shift")?.parse()?)
            }
            "--output-backward-shift" => {
                config.output_backward_shift =
                    Some(next(&mut args, "--output-backward-shift")?.parse()?)
            }
            "--probability-gradient-fractional-bits" => {
                config.probability_gradient_fractional_bits =
                    next(&mut args, "--probability-gradient-fractional-bits")?.parse()?
            }
            "--probability-normalization" => {
                let value = next(&mut args, "--probability-normalization")?;
                config.probability_normalization = match value.as_str() {
                    "legacy-q31-lut" | "legacy_q31_lut" => SoftmaxNormalization::LegacyQ31Lut,
                    "q47-lut" | "q47_lut" => SoftmaxNormalization::Q47Lut,
                    "q47-newton1" | "q47_newton1" => SoftmaxNormalization::Q47Newton1,
                    "q47-exact-division" | "q47_exact_division" => SoftmaxNormalization::Q47Exact,
                    _ => {
                        return Err(
                            format!("unsupported probability normalization: {value}").into()
                        );
                    }
                }
            }
            "--batch-windows" => {
                config.batch_windows = next(&mut args, "--batch-windows")?.parse()?
            }
            "--max-optimizer-steps" => {
                config.max_optimizer_steps = next(&mut args, "--max-optimizer-steps")?.parse()?
            }
            "--evaluation-windows" => {
                config.evaluation_windows = next(&mut args, "--evaluation-windows")?.parse()?
            }
            "--reject-saturated-batch" => config.reject_saturated_batch = true,
            "--flush-batched-embedding-residuals" => {
                config.flush_batched_embedding_residuals = true
            }
            "--descent-guard-windows" => {
                config.descent_guard_windows =
                    next(&mut args, "--descent-guard-windows")?.parse()?
            }
            "--descent-guard-signed-representation-blocks" => {
                config.descent_guard_signed_representation_blocks = true
            }
            "--descent-guard-signed-representation-zero-saturation" => {
                config.descent_guard_signed_representation_zero_saturation = true
            }
            "--backward-quantization" => {
                let value = next(&mut args, "--backward-quantization")?;
                config.backward_quantization = match value.as_str() {
                    "rescued-rhu" | "rescued_rhu" => ProductionBackwardQuantization::RescuedRhu,
                    "late-rhu" | "late_rhu" => ProductionBackwardQuantization::LateRhu,
                    "late-stochastic" | "late_stochastic" => {
                        ProductionBackwardQuantization::LateStochastic
                    }
                    _ => return Err(format!("unsupported backward quantization: {value}").into()),
                }
            }
            "--backward-stochastic-seed" => {
                config.backward_stochastic_seed =
                    next(&mut args, "--backward-stochastic-seed")?.parse()?
            }
            "--coordinates-per-group" => {
                config.alignment_coordinates_per_group =
                    next(&mut args, "--coordinates-per-group")?.parse()?
            }
            "--transfer-windows" | "--acceptance-windows" => {
                config.alignment_transfer_windows =
                    next(&mut args, "--transfer-windows")?.parse()?
            }
            "--documents-per-surface" => {
                config.alignment_documents_per_surface =
                    next(&mut args, "--documents-per-surface")?.parse()?
            }
            "--rescue-stratified-sampling" => config.alignment_rescue_stratified_sampling = true,
            "--include-mass-corrected-no-rescue" => {
                config.alignment_include_mass_corrected_no_rescue = true
            }
            "--include-systematic-fixed-mass" => {
                config.alignment_include_systematic_fixed_mass = true
            }
            "--expected-trunk-moves" => {
                config.boolean_jet_expected_trunk_moves =
                    next(&mut args, "--expected-trunk-moves")?.parse()?
            }
            "--expected-head-moves" => {
                config.boolean_jet_expected_head_moves =
                    next(&mut args, "--expected-head-moves")?.parse()?
            }
            "--expected-move-fingerprint" => {
                config.boolean_jet_expected_move_fingerprint =
                    parse_u64_literal(&next(&mut args, "--expected-move-fingerprint")?)?
            }
            "--expected-manifest-hash" => {
                config.boolean_jet_expected_manifest_hash =
                    parse_u64_literal(&next(&mut args, "--expected-manifest-hash")?)?
            }
            "--expected-base-model-hash" => {
                config.boolean_jet_expected_base_model_hash =
                    parse_u64_literal(&next(&mut args, "--expected-base-model-hash")?)?
            }
            "--expected-tokenizer-hash" => {
                config.boolean_jet_expected_tokenizer_hash =
                    parse_u64_literal(&next(&mut args, "--expected-tokenizer-hash")?)?
            }
            "--expected-token-stream-hash" => {
                config.boolean_jet_expected_token_stream_hash =
                    parse_u64_literal(&next(&mut args, "--expected-token-stream-hash")?)?
            }
            "--expected-source-fnv64" => {
                config.boolean_jet_expected_source_fnv64 =
                    parse_u64_literal(&next(&mut args, "--expected-source-fnv64")?)?
            }
            "--expected-binary-fnv64" => {
                config.boolean_jet_expected_binary_fnv64 =
                    parse_u64_literal(&next(&mut args, "--expected-binary-fnv64")?)?
            }
            "--trunk-move" => config
                .boolean_jet_trunk_moves
                .push(parse_boolean_jet_move(&next(&mut args, "--trunk-move")?)?),
            "--head-move" => config
                .boolean_jet_head_moves
                .push(parse_boolean_jet_move(&next(&mut args, "--head-move")?)?),
            "--control-move" => config
                .boolean_jet_control_moves
                .push(parse_boolean_jet_move(&next(&mut args, "--control-move")?)?),
            "--proposal-document-start" => {
                config.boolean_jet_proposal_document_start =
                    next(&mut args, "--proposal-document-start")?.parse()?
            }
            "--proposal-documents" => {
                config.boolean_jet_proposal_documents =
                    next(&mut args, "--proposal-documents")?.parse()?
            }
            "--transfer-document-start" => {
                config.boolean_jet_transfer_document_start =
                    next(&mut args, "--transfer-document-start")?.parse()?
            }
            "--transfer-documents" => {
                config.boolean_jet_transfer_documents =
                    next(&mut args, "--transfer-documents")?.parse()?
            }
            "--windows-per-document" => {
                config.boolean_jet_windows_per_document =
                    next(&mut args, "--windows-per-document")?.parse()?
            }
            "--minimum-documents" => {
                config.boolean_jet_minimum_documents =
                    next(&mut args, "--minimum-documents")?.parse()?
            }
            "--reserved-document-start" => {
                config.boolean_jet_reserved_document_start =
                    next(&mut args, "--reserved-document-start")?.parse()?
            }
            "--help" | "-h" => config.command = "help".to_string(),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

fn next(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn parse_u64_literal(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(hex) = value.strip_prefix("0x") {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse()?)
    }
}

fn parse_boolean_jet_move(value: &str) -> Result<BooleanJetMoveArg, Box<dyn std::error::Error>> {
    let mut parts = value.split(':');
    let group_index = parts
        .next()
        .ok_or("Boolean-jet move requires GROUP:COORDINATE:DELTA")?
        .parse()?;
    let coordinate = parts
        .next()
        .ok_or("Boolean-jet move requires GROUP:COORDINATE:DELTA")?
        .parse()?;
    let delta = parts
        .next()
        .ok_or("Boolean-jet move requires GROUP:COORDINATE:DELTA")?
        .parse()?;
    if parts.next().is_some() || !matches!(delta, -1 | 1) {
        return Err("Boolean-jet move requires GROUP:COORDINATE:DELTA with unit DELTA".into());
    }
    Ok(BooleanJetMoveArg {
        group_index,
        coordinate,
        delta,
    })
}

fn required(value: Option<PathBuf>, option: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("{option} is required").into())
}

fn print_help() {
    println!(
        "Smoke-training sampling:\n  pass --spread-windows to select deterministic uniformly spaced target windows across the complete document stream. Pass --targets-per-window N on full-train-smoke to supervise the causal suffix of each context with an averaged multi-target objective. Pass --training-workers N to parallelize deterministic output-head work without changing the schedule or trained bytes. The schedule-bound --embedding-learning-rate-boost-shift N option permits an embedding learning rate above the fixed-point mean-shift floor. Pass --flush-batched-embedding-residuals to apply accumulated embedding residuals for every token touched anywhere in a committed batch. Pass --descent-guard-windows N to reject moving batches that worsen canonical NLL on N fixed, disjoint training-corpus windows while still consuming the batch cursor. Add --descent-guard-signed-representation-blocks to search reverse/zero/forward steps for embeddings, K, V, and O, choosing the non-worsening minimum with a sparse deterministic tie-break. Add --descent-guard-signed-representation-zero-saturation to discard candidates with any forward residual saturation on that training-only guard before applying the NLL tie-break. Pass --output-bias-learning-rate-shift N to control output-bias updates independently from RMS vectors. Pass --backward-quantization rescued-rhu|late-rhu|late-stochastic to select the schedule-bound intermediate-gradient quantizer; stochastic mode also binds --backward-stochastic-seed N for exact restart replay.\n"
    );
    println!(
        "Additional audit:\n  nsrl-production-model boolean-jet-rank-two-audit --tokenizer PATH --tokens PATH --model PATH --trace PATH [--expected-trunk-moves N] [--expected-head-moves N] [--expected-move-fingerprint U64] [gradient-alignment and training numeric options]\n"
    );
    println!(
        "Hardened Boolean-jet schema:\n  use boolean-jet-audit with the same arguments to emit nsrl.production_boolean_jet.v1; the rank-two command preserves the frozen legacy artifact schema.\n"
    );
    println!(
        "Prospective confirmation:\n  nsrl-production-model boolean-jet-confirmation-audit --tokenizer PATH --tokens PATH --model PATH --trace PATH --expected-base-model-hash U64 --expected-tokenizer-hash U64 --expected-token-stream-hash U64 --expected-move-fingerprint U64 --expected-manifest-hash U64 --trunk-move GROUP:COORDINATE:DELTA... --head-move GROUP:COORDINATE:DELTA... --control-move GROUP:COORDINATE:DELTA... [--proposal-document-start N] [--proposal-documents N] [--transfer-document-start N] [--transfer-documents N] [--windows-per-document N] [--minimum-documents N]\n"
    );
    println!(
        "Usage:\n  nsrl-production-model init --profile p10m|p20m|p30m --tokenizer PATH --model-out PATH --trace PATH [--seed N] [--output-init-amplitude N] [--output-forward-shift N] [--up-forward-shift N]\n  nsrl-production-model inspect --model PATH\n  nsrl-production-model numeric-contract [--profile p10m|p20m|p30m | --model PATH] [--trace PATH] [--output-forward-shift N] [--up-forward-shift N]\n  nsrl-production-model generate --tokenizer PATH --model PATH (--prompt TEXT | --prompt-file PATH) --trace PATH [--generated-out PATH] [--context-tokens N] [--max-new-tokens N] [--top-k N] [--seed N] [--no-stop-on-eos]\n  nsrl-production-model evaluate --tokenizer PATH --tokens PATH --model PATH --trace PATH [--context-tokens N] [--max-windows N]\n  nsrl-production-model evaluate-canonical --tokenizer PATH --tokens PATH --model PATH --trace PATH [--context-tokens N] [--max-windows N]\n  nsrl-production-model gradient-alignment-audit --tokenizer PATH --tokens PATH --model PATH --trace PATH [--context-tokens N] [--max-windows N] [--transfer-windows N] [--documents-per-surface N] [--rescue-stratified-sampling] [--include-mass-corrected-no-rescue] [--coordinates-per-group N] [--seed N] [training numeric options]\n  nsrl-production-model compare-evaluate --tokenizer PATH --tokens PATH --model SOURCE --candidate-model CANDIDATE --trace PATH [--context-tokens N] [--max-windows N] [--up-forward-shift N]\n  nsrl-production-model probability-resolution-audit --tokenizer PATH --tokens PATH --model SOURCE --candidate-model CANDIDATE --trace PATH [--context-tokens N] [--max-windows N] [--up-forward-shift N]\n  nsrl-production-model probability-normalization-audit --tokenizer PATH --tokens PATH --model SOURCE --candidate-model CANDIDATE --trace PATH [--context-tokens N] [--max-windows N] [--up-forward-shift N]\n  nsrl-production-model probability-normalization-signal-attribution-audit --tokenizer PATH --tokens PATH --model SOURCE --candidate-model CANDIDATE --trace PATH [--context-tokens N] [--max-windows N] [--up-forward-shift N]\n  nsrl-production-model smoke-train --tokenizer PATH --tokens PATH --model PATH --model-out PATH --trace PATH [--context-tokens N] [--max-windows N] [--epochs N] [--feature-shift N] [--bias-step-q8 N] [--margin-q8 N]\n  nsrl-production-model target-margin-train --tokenizer PATH --tokens PATH --model PATH --model-out PATH --optimizer-state-out PATH --trace PATH [--optimizer-state PATH] [--context-tokens N] [--targets-per-window N] [--training-workers N] [--spread-windows] [--max-windows N] [--evaluation-windows N] [--epochs N] [--feature-shift N] [--margin-q8 N] [--batch-windows N] [--max-optimizer-steps N]\n  nsrl-production-model full-train-smoke --tokenizer PATH --tokens PATH --model PATH --model-out PATH --optimizer-state-out PATH --trace PATH [--optimizer-state PATH] [--context-tokens N] [--targets-per-window N] [--spread-windows] [--max-windows N] [--evaluation-windows N] [--epochs N] [--batch-windows N] [--max-optimizer-steps N] [--reject-saturated-batch] [--matrix-learning-rate-shift N] [--q-learning-rate-shift N] [--k-learning-rate-shift N] [--v-learning-rate-shift N] [--o-learning-rate-shift N] [--up-learning-rate-shift N] [--gate-learning-rate-shift N] [--down-learning-rate-shift N] [--vector-learning-rate-shift N] [--embedding-learning-rate-shift N] [--embedding-learning-rate-boost-shift N] [--output-learning-rate-shift N] [--output-backward-shift N] [--probability-gradient-fractional-bits 15..31] [--probability-normalization legacy-q31-lut|q47-lut|q47-newton1|q47-exact-division]
  nsrl-production-model direct-head-train --tokenizer PATH --tokens PATH --model PATH --model-out PATH --trace PATH [--context-tokens N] [--max-windows N] [--evaluation-windows N] [--max-optimizer-steps N] [--coordinates-per-group N] [--probability-gradient-fractional-bits 15..31] [--probability-normalization legacy-q31-lut|q47-lut|q47-newton1|q47-exact-division] [--seed N]
  nsrl-production-model direct-feature-train --tokenizer PATH --tokens PATH --model PATH --model-out PATH --trace PATH [--context-tokens N] [--max-windows N] [--evaluation-windows N] [--max-optimizer-steps N] [--coordinates-per-group N] [--probability-gradient-fractional-bits 15..31] [--probability-normalization legacy-q31-lut|q47-lut|q47-newton1|q47-exact-division] [--seed N]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> impl Iterator<Item = String> {
        values.iter().map(|value| (*value).to_string())
    }

    #[test]
    fn generation_arguments_are_distinct_from_training_limits() {
        let config = parse_args(args(&[
            "generate",
            "--tokenizer",
            "tokenizer.nsrlbpe",
            "--model",
            "model.nsrlpm",
            "--prompt",
            "The king",
            "--trace",
            "trace.json",
            "--generated-out",
            "continuation.txt",
            "--context-tokens",
            "128",
            "--max-new-tokens",
            "512",
            "--top-k",
            "32",
            "--seed",
            "19",
            "--no-stop-on-eos",
        ]))
        .expect("generation arguments");
        assert_eq!(config.command, "generate");
        assert_eq!(config.prompt.as_deref(), Some("The king"));
        assert_eq!(config.context_tokens, 128);
        assert!(config.context_tokens_explicit);
        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.top_k, 32);
        assert_eq!(config.seed, 19);
        assert!(!config.stop_on_eos);
        assert_eq!(
            config.max_windows,
            ProductionSmokeConfig::default().max_windows
        );
    }

    #[test]
    fn spread_window_flag_is_available_to_training_commands() {
        let config = parse_args(args(&[
            "full-train-smoke",
            "--tokenizer",
            "tokenizer.nsrlbpe",
            "--tokens",
            "train.nsrltok",
            "--model",
            "model.nsrlpm",
            "--model-out",
            "trained.nsrlpm",
            "--optimizer-state-out",
            "optimizer.nsrlpo",
            "--trace",
            "trace.json",
            "--spread-windows",
            "--targets-per-window",
            "4",
            "--training-workers",
            "3",
            "--output-bias-learning-rate-shift",
            "14",
            "--embedding-learning-rate-boost-shift",
            "1",
            "--flush-batched-embedding-residuals",
            "--descent-guard-windows",
            "64",
            "--descent-guard-signed-representation-blocks",
            "--descent-guard-signed-representation-zero-saturation",
            "--backward-quantization",
            "late-stochastic",
            "--backward-stochastic-seed",
            "29",
        ]))
        .expect("spread training arguments");
        assert!(config.spread_windows);
        assert_eq!(config.targets_per_window, 4);
        assert_eq!(config.training_workers, 3);
        assert_eq!(config.output_bias_learning_rate_shift, Some(14));
        assert_eq!(config.embedding_learning_rate_boost_shift, 1);
        assert!(config.flush_batched_embedding_residuals);
        assert_eq!(config.descent_guard_windows, 64);
        assert!(config.descent_guard_signed_representation_blocks);
        assert!(config.descent_guard_signed_representation_zero_saturation);
        assert_eq!(
            config.backward_quantization,
            ProductionBackwardQuantization::LateStochastic
        );
        assert_eq!(config.backward_stochastic_seed, 29);
    }

    #[test]
    fn help_is_a_parseable_command() {
        assert_eq!(parse_args(args(&["help"])).expect("help").command, "help");
    }
}
