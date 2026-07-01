#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::{
    ByteTokenizerId, MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HEADS,
    MiniTransformerAttentionKind, MiniTransformerBatchMode, MiniTransformerMlpModel,
    MiniTransformerMlpTrainConfig, MiniTransformerNextTokenRow, MiniTransformerPositionPolicy,
    MiniTransformerTargetSegment, MiniTransformerTraceDetail,
    mini_transformer_next_token_row_with_attention_kind_position_policy,
    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail,
};

const TRAIN_SCHEMA: &str = "nsrl.solomon_attention_train_trace.v1";
const SAMPLE_SCHEMA: &str = "nsrl.solomon_attention_sample_trace.v1";
const EVAL_SCHEMA: &str = "nsrl.solomon_attention_eval_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLLMM1";
const MODEL_VERSION: u32 = 4;

const PAD: u8 = 0;
const BOS: u8 = 1;
const PROMPT: u8 = 2;
const TEXT: u8 = 3;
const IMAGE: u8 = 4;
const EOS: u8 = 5;
const TEXT_BASE: u8 = 16;
const TEXT_COUNT: u8 = 128;
const IMAGE_BASE: u8 = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS: u8 = 16;
const TEXT_CHUNK_BASE: u8 = 160;
const TEXT_CHUNK_NAME_START: usize = 24;
const VOCAB_SIZE: usize = 256;
const LEGACY_VOCAB_SIZE: u32 = IMAGE_BASE as u32 + IMAGE_BINS as u32;
const VOCAB_LEN: usize = VOCAB_SIZE;
const SIGNATURE_GRID: usize = 16;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
const NAME_OPENING_PREFIX: &str = "Solomon selects ";

const TEXT_CHUNKS: &[&str] = &[
    "Solomon selects ",
    ": ",
    "He ",
    "is ",
    "appeareth ",
    "maketh ",
    "teacheth ",
    "giveth ",
    "causeth ",
    "knoweth ",
    "healeth ",
    "teaches ",
    "and ",
    "the ",
    "of ",
    "to ",
    "in ",
    "a ",
    "his ",
    "with ",
    "upon ",
    "unto ",
    "This ",
    "His ",
    "Bael",
    "Agares",
    "Vassago",
    "Samigina",
    "Marbas",
    "Valefor",
    "Amon",
    "Barbatos",
    "Paimon",
    "Buer",
    "Gusion",
    "Sitri",
    "Beleth",
    "Leraje",
    "Eligos",
    "Zepar",
    "Botis",
    "Bathin",
    "Sallos",
    "Purson",
    "Marax",
    "Ipos",
    "Aim",
    "Naberius",
    "Glasya-Labolas",
    "Bune",
    "Ronove",
    "Berith",
    "Astaroth",
    "Forneus",
    "Foras",
    "Asmoday",
    "Gaap",
    "Furfur",
    "Marchosias",
    "Stolas",
    "Phenex",
    "Halphas",
    "Malphas",
    "Raum",
    "Focalor",
    "Vepar",
    "Sabnock",
    "Shax",
    "Vine",
    "Bifrons",
    "Uvall",
    "Haagenti",
    "Crocell",
    "Furcas",
    "Balam",
    "Alloces",
    "Camio",
    "Murmur",
    "Orobas",
    "Gremory",
    "Ose",
    "Amy",
    "Oriax",
    "Vapula",
    "Zagan",
    "Volac",
    "Andras",
    "Haures",
    "Andrealphus",
    "Cimejes",
    "Amdusias",
    "Belial",
    "Decarabia",
    "Seere",
    "Dantalion",
    "Andromalius",
];

const DEFAULT_EPOCHS: usize = 1;
const DEFAULT_SEQ_LEN: usize = 64;
const DEFAULT_STRIDE: usize = 1;
const DEFAULT_MAX_WINDOWS: usize = 1024;
const DEFAULT_BATCH_WINDOWS: usize = 8;
const DEFAULT_MIN_TEXT_TOKENS: usize = 12;
const DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8: i32 = 1_000_000;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Train,
    Sample,
    Eval,
}

#[derive(Debug, Clone)]
struct Config {
    command: Command,
    tokens_path: PathBuf,
    init_model: Option<PathBuf>,
    embed_text_memory_examples: Option<PathBuf>,
    embed_text_memory_order: usize,
    model_path: PathBuf,
    model_out: PathBuf,
    out_dir: PathBuf,
    prompt: String,
    text_prefix: String,
    epochs: usize,
    seq_len: usize,
    stride: usize,
    window_offset: usize,
    max_windows: Option<usize>,
    batch_windows: usize,
    target_token_min: u8,
    target_token_max: u8,
    target_segment: TargetSegment,
    target_frequency_cap: u32,
    target_frequency_min_weight_q15: i16,
    argmax_margin_weight_q15: i16,
    zero_output_head_init: bool,
    solomon_name_copy_init: bool,
    solomon_name_copy_repair: bool,
    solomon_name_copy_repair_preserve_body_output: bool,
    solomon_body_scaffold: bool,
    solomon_body_opening_repair: bool,
    progress_interval_batches: usize,
    learning_rate: i32,
    output_learning_rate_shift: u8,
    mlp_learning_rate_shift: u8,
    embedding_learning_rate_shift: u8,
    attention_learning_rate_shift: u8,
    attention_q_learning_rate_shift: u8,
    attention_qk_learning_rate_shift: u8,
    reject_loss_regression: bool,
    max_text_tokens: usize,
    min_text_tokens: usize,
    repeat_run_cap: usize,
    no_repeat_ngram: usize,
    conditioning_examples: Option<PathBuf>,
    conditioning_boost_q8: i32,
    text_prior_examples: Option<PathBuf>,
    text_prior_order: usize,
    text_prior_min_order: usize,
    text_prior_boost_q8: i32,
    text_prior_strict: bool,
    use_embedded_text_memory: bool,
    embedded_text_lm_order: usize,
    text_chunk_boost_q8: i32,
    decode_logit_delta: bool,
    prompt_name_opening_prior: bool,
    suppress_name_chunks_after_opening: bool,
    top_k: usize,
    sample_seed: u64,
    eval_max_examples: Option<usize>,
    text_token_profile: TextTokenProfile,
    position_policy: MiniTransformerPositionPolicy,
    adaptive_attention_shifts: bool,
    attention_vo_oracle: bool,
}

impl Default for Config {
    fn default() -> Self {
        let root = PathBuf::from("data/processed/key-solomon-goetia-attention-v1");
        Self {
            command: Command::Train,
            tokens_path: root.join("corpus.tokens.u8"),
            init_model: None,
            embed_text_memory_examples: None,
            embed_text_memory_order: 32,
            model_path: root.join("model.nsrllmm"),
            model_out: root.join("model.nsrllmm"),
            out_dir: root.join("attention-sample"),
            prompt: String::from("king solomon seal"),
            text_prefix: String::new(),
            epochs: DEFAULT_EPOCHS,
            seq_len: DEFAULT_SEQ_LEN,
            stride: DEFAULT_STRIDE,
            window_offset: 0,
            max_windows: Some(DEFAULT_MAX_WINDOWS),
            batch_windows: DEFAULT_BATCH_WINDOWS,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: TargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: 4096,
            argmax_margin_weight_q15: 0,
            zero_output_head_init: false,
            solomon_name_copy_init: false,
            solomon_name_copy_repair: false,
            solomon_name_copy_repair_preserve_body_output: false,
            solomon_body_scaffold: false,
            solomon_body_opening_repair: false,
            progress_interval_batches: 0,
            learning_rate: 1,
            output_learning_rate_shift: 18,
            mlp_learning_rate_shift: 16,
            embedding_learning_rate_shift: 14,
            attention_learning_rate_shift: 24,
            attention_q_learning_rate_shift: 18,
            attention_qk_learning_rate_shift: 18,
            reject_loss_regression: false,
            max_text_tokens: 192,
            min_text_tokens: DEFAULT_MIN_TEXT_TOKENS,
            repeat_run_cap: 8,
            no_repeat_ngram: 4,
            conditioning_examples: Some(root.join("examples.jsonl")),
            conditioning_boost_q8: 1_000_000,
            text_prior_examples: None,
            text_prior_order: 8,
            text_prior_min_order: 0,
            text_prior_boost_q8: 0,
            text_prior_strict: false,
            use_embedded_text_memory: true,
            embedded_text_lm_order: 0,
            text_chunk_boost_q8: 0,
            decode_logit_delta: false,
            prompt_name_opening_prior: false,
            suppress_name_chunks_after_opening: false,
            top_k: 1,
            sample_seed: 1,
            eval_max_examples: Some(16),
            text_token_profile: TextTokenProfile::Char,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            adaptive_attention_shifts: false,
            attention_vo_oracle: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SolomonAttentionModel {
    token_count: u64,
    token_hash: u64,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    text_token_profile: TextTokenProfile,
    text_memory: Option<TextMemory>,
    transformer: MiniTransformerMlpModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sample {
    prompt: String,
    text: String,
    image_bins: [u8; SIGNATURE_BINS],
    generated_tokens: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleOptions {
    max_text_tokens: usize,
    min_text_tokens: usize,
    repeat_run_cap: usize,
    no_repeat_ngram: usize,
    conditioning_boost_q8: i32,
    text_prior_boost_q8: i32,
    text_prior_strict: bool,
    text_chunk_boost_q8: i32,
    decode_logit_delta: bool,
    prompt_name_opening_prior: bool,
    suppress_name_chunks_after_opening: bool,
    top_k: usize,
    sample_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Text,
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextTokenProfile {
    Char,
    Chunked,
}

impl TextTokenProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Chunked => "chunked",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetSegment {
    All,
    GeneratedText,
    NameOpening,
    NameOpeningTail,
    BodyAfterHe,
    BodyFirstAfterHe,
    BodyFirstAfterOpening,
    Image,
}

impl TargetSegment {
    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::GeneratedText => "generated-text",
            Self::NameOpening => "name-opening",
            Self::NameOpeningTail => "name-opening-tail",
            Self::BodyAfterHe => "body-after-he",
            Self::BodyFirstAfterHe => "body-first-after-he",
            Self::BodyFirstAfterOpening => "body-first-after-opening",
            Self::Image => "image",
        }
    }

    fn to_train_segment(
        self,
        text_token_profile: TextTokenProfile,
    ) -> MiniTransformerTargetSegment {
        match self {
            Self::All => MiniTransformerTargetSegment::All,
            Self::GeneratedText => {
                MiniTransformerTargetSegment::after_marker_before_any(TEXT, &[IMAGE, EOS])
                    .expect("valid generated-text target segment")
            }
            Self::NameOpening => MiniTransformerTargetSegment::after_sequence_before_any(
                &name_opening_start_sequence(text_token_profile),
                &name_opening_end_markers(text_token_profile),
            )
            .expect("valid name-opening target segment"),
            Self::NameOpeningTail => MiniTransformerTargetSegment::after_sequence_before_any(
                &name_opening_start_sequence(text_token_profile),
                &[IMAGE, EOS],
            )
            .expect("valid name-opening-tail target segment"),
            Self::BodyAfterHe => body_after_he_train_segment(text_token_profile),
            Self::BodyFirstAfterHe => body_first_after_he_train_segment(text_token_profile),
            Self::BodyFirstAfterOpening => {
                body_first_after_opening_train_segment(text_token_profile)
            }
            Self::Image => MiniTransformerTargetSegment::after_marker_before_any(IMAGE, &[EOS])
                .expect("valid image target segment"),
        }
    }
}

fn name_opening_start_sequence(text_token_profile: TextTokenProfile) -> Vec<u8> {
    std::iter::once(TEXT)
        .chain(encode_text_prefix_tokens(
            NAME_OPENING_PREFIX,
            text_token_profile,
        ))
        .collect()
}

fn name_opening_end_markers(text_token_profile: TextTokenProfile) -> Vec<u8> {
    match text_token_profile {
        TextTokenProfile::Char => vec![TEXT_BASE + b':' as u8, IMAGE, EOS],
        TextTokenProfile::Chunked => vec![TEXT_CHUNK_BASE + 1, IMAGE, EOS],
    }
}

fn body_after_he_train_segment(
    text_token_profile: TextTokenProfile,
) -> MiniTransformerTargetSegment {
    match text_token_profile {
        TextTokenProfile::Char => MiniTransformerTargetSegment::after_sequence_before_any(
            &encode_text_prefix_tokens("He ", text_token_profile),
            &[IMAGE, EOS],
        )
        .expect("valid body-after-he target segment"),
        TextTokenProfile::Chunked => MiniTransformerTargetSegment::after_marker_before_any(
            TEXT_CHUNK_BASE + 2,
            &[IMAGE, EOS],
        )
        .expect("valid body-after-he target segment"),
    }
}

fn body_first_after_he_train_segment(
    text_token_profile: TextTokenProfile,
) -> MiniTransformerTargetSegment {
    MiniTransformerTargetSegment::first_after_sequence_before_any(
        &encode_text_prefix_tokens("He ", text_token_profile),
        &[IMAGE, EOS],
    )
    .expect("valid body-first-after-he target segment")
}

fn body_first_after_opening_train_segment(
    text_token_profile: TextTokenProfile,
) -> MiniTransformerTargetSegment {
    MiniTransformerTargetSegment::first_after_sequence_before_any(
        &encode_text_prefix_tokens(": ", text_token_profile),
        &[IMAGE, EOS],
    )
    .expect("valid body-first-after-opening target segment")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    token: u8,
    logit_q8: i32,
    probability_q15: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodeOptions {
    seed: u64,
    repeat_run_cap: usize,
    no_repeat_ngram: usize,
    expected_token_boost_q8: i32,
    text_prior_boost_q8: i32,
    text_prior_strict: bool,
    text_chunk_boost_q8: i32,
    decode_logit_delta: bool,
    prompt_name_opening_tokens: Vec<u8>,
    suppress_name_chunks_after_opening: bool,
    top_k: usize,
}

#[derive(Debug, Clone, Copy)]
struct NextTokenRequest<'a> {
    history: &'a [u8],
    phase: Phase,
    allow_stop: bool,
    step: usize,
    text_token_profile: TextTokenProfile,
    expected_token: Option<u8>,
    text_prior: Option<&'a TextPrior>,
    text_prior_start_tokens: Option<&'a [u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConditioningMatch {
    primary_name: String,
    prompt: String,
    score: usize,
    text_tokens: Vec<u8>,
    image_tokens: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextPrior {
    order: usize,
    min_order: usize,
    start_tokens: Vec<u8>,
    prompt_starts: Vec<PromptTextStart>,
    transitions: HashMap<Vec<u8>, [u32; VOCAB_LEN]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptTextStart {
    prompt_key: String,
    primary_key: String,
    tokens: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextMemory {
    order: usize,
    examples: Vec<TextMemoryExample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextMemoryExample {
    primary_name: String,
    prompt: String,
    text_tokens: Vec<u8>,
    image_tokens: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalPhase {
    Special,
    Prompt,
    Text,
    Image,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EvalStats {
    targets: usize,
    correct: usize,
    invalid_contexts: usize,
    probability_error_q15: u64,
    target_rank_sum: u64,
    top5_correct: usize,
    top10_correct: usize,
    target_margin_q8_sum: i64,
    target_margin_q8_min: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvalTokenDiagnostics {
    predicted: u8,
    target_rank: usize,
    target_margin_q8: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExampleMarkers {
    prompt: usize,
    text: usize,
    image: usize,
    eos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PaddedContext {
    storage: Vec<u8>,
}

impl PaddedContext {
    fn new(context_seq_len: usize) -> Self {
        Self {
            storage: Vec::with_capacity(context_seq_len),
        }
    }

    fn window<'a>(&'a mut self, history: &'a [u8], context_seq_len: usize) -> &'a [u8] {
        let context_len = context_seq_len.min(history.len());
        let start = history.len().saturating_sub(context_len);
        if context_len < context_seq_len {
            self.storage.clear();
            self.storage.resize(context_seq_len - context_len, PAD);
            self.storage.extend_from_slice(&history[start..]);
            self.storage.as_slice()
        } else {
            &history[start..]
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-solomon-attention: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    match config.command {
        Command::Train => train_command(config),
        Command::Sample => sample_command(config),
        Command::Eval => eval_command(config),
    }
}

fn usage() {
    println!(
        "Usage: nsrl-solomon-attention train|sample|eval [--tokens PATH] [--model PATH] [--model-out PATH]\n\
         \n\
         Train options: [--epochs N] [--seq-len N] [--stride N] [--window-offset N]\n\
         \t[--max-windows N|none]\n\
         \t[--init-model PATH]\n\
         \t[--embed-text-memory-examples PATH] [--embed-text-memory-order N]\n\
         \t[--text-token-profile char|chunked]\n\
         \t[--position-policy learned-absolute|nope]\n\
         \t[--learning-rate N]\n\
         \t[--output-lr-shift N] [--mlp-lr-shift N] [--embed-lr-shift N]\n\
         \t[--attention-lr-shift N] [--attention-q-lr-shift N] [--attention-qk-lr-shift N]\n\
         \t[--target-token-range MIN MAX] [--target-phase all|special|text-char|text-chunk|image]\n\
         \t[--target-segment all|generated-text|name-opening|name-opening-tail|body-after-he|body-first-after-he|body-first-after-opening|image]\n\
         \t[--target-frequency-cap N] [--target-frequency-min-weight-q15 N]\n\
         \t[--argmax-margin-weight-q15 N]\n\
         \t[--zero-output-head-init] [--solomon-name-copy-init] [--solomon-name-copy-repair]\n\
         \t[--solomon-name-copy-repair-preserve-body-output]\n\
         \t[--solomon-body-scaffold]\n\
         \t[--reject-loss-regression]\n\
         \t[--batch-windows N] [--progress-interval-batches N]\n\
         \t[--adaptive-attention-shifts] [--attention-vo-oracle]\n\
         Sample options: [--out-dir PATH] [--prompt TEXT] [--text-prefix TEXT]\n\
         \t[--max-text-tokens N]\n\
         \t[--min-text-tokens N] [--repeat-run-cap N] [--no-repeat-ngram N]\n\
         \t[--conditioning-examples PATH|none]\n\
         \t[--conditioning-boost-q8 N] [--text-prior-examples PATH|none]\n\
         \t[--text-prior-order N] [--text-prior-min-order N]\n\
         \t[--text-prior-boost-q8 N] [--text-prior-strict]\n\
         \t[--no-embedded-text-memory] [--embedded-text-lm-order N]\n\
         \t[--text-chunk-boost-q8 N]\n\
         \t[--decode-logit-delta] [--prompt-name-opening-prior]\n\
         \t[--suppress-name-chunks-after-opening]\n\
         \t[--top-k N] [--sample-seed N]\n\
         Eval options: [--tokens PATH] [--model PATH] [--conditioning-examples PATH]\n\
         \t[--eval-max-examples N|none]"
    );
}

fn parse_args<I>(args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let mut config = Config::default();
    let mut args = args.peekable();
    if let Some(arg) = args.peek()
        && !arg.starts_with("--")
    {
        config.command = parse_command(&args.next().unwrap())?;
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--tokens" => {
                config.tokens_path = PathBuf::from(args.next().ok_or("--tokens requires PATH")?);
            }
            "--init-model" => {
                config.init_model = Some(PathBuf::from(
                    args.next().ok_or("--init-model requires PATH")?,
                ));
            }
            "--embed-text-memory-examples" => {
                config.embed_text_memory_examples = Some(PathBuf::from(
                    args.next()
                        .ok_or("--embed-text-memory-examples requires PATH")?,
                ));
            }
            "--embed-text-memory-order" => {
                config.embed_text_memory_order =
                    parse_usize(args.next(), "--embed-text-memory-order")?;
            }
            "--model" => {
                config.model_path = PathBuf::from(args.next().ok_or("--model requires PATH")?);
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
            }
            "--prompt" => {
                config.prompt = args.next().ok_or("--prompt requires TEXT")?;
            }
            "--text-prefix" => {
                config.text_prefix = args.next().ok_or("--text-prefix requires TEXT")?;
            }
            "--epochs" => {
                config.epochs = parse_positive_usize(args.next(), "--epochs")?;
            }
            "--seq-len" => {
                config.seq_len = parse_positive_usize(args.next(), "--seq-len")?;
            }
            "--stride" => {
                config.stride = parse_positive_usize(args.next(), "--stride")?;
            }
            "--window-offset" => {
                config.window_offset = parse_usize(args.next(), "--window-offset")?;
            }
            "--max-windows" => {
                let value = args.next().ok_or("--max-windows requires N or none")?;
                config.max_windows = if value == "none" {
                    None
                } else {
                    Some(parse_positive_usize(Some(value), "--max-windows")?)
                };
            }
            "--batch-windows" => {
                config.batch_windows = parse_positive_usize(args.next(), "--batch-windows")?;
            }
            "--target-token-range" => {
                config.target_token_min = parse_u8(args.next(), "--target-token-range MIN")?;
                config.target_token_max = parse_u8(args.next(), "--target-token-range MAX")?;
            }
            "--target-phase" => {
                let phase = args.next().ok_or(
                    "--target-phase requires all, special, text-char, text-chunk, or image",
                )?;
                (config.target_token_min, config.target_token_max) = parse_target_phase(&phase)?;
            }
            "--target-segment" => {
                let segment = args.next().ok_or(
                    "--target-segment requires all, generated-text, name-opening, name-opening-tail, body-after-he, body-first-after-he, body-first-after-opening, or image",
                )?;
                config.target_segment = parse_target_segment(&segment)?;
            }
            "--target-frequency-cap" => {
                config.target_frequency_cap = args
                    .next()
                    .ok_or("--target-frequency-cap requires N")?
                    .parse()?;
            }
            "--target-frequency-min-weight-q15" => {
                config.target_frequency_min_weight_q15 = args
                    .next()
                    .ok_or("--target-frequency-min-weight-q15 requires N")?
                    .parse()?;
            }
            "--argmax-margin-weight-q15" => {
                config.argmax_margin_weight_q15 = args
                    .next()
                    .ok_or("--argmax-margin-weight-q15 requires N")?
                    .parse()?;
            }
            "--zero-output-head-init" => {
                config.zero_output_head_init = true;
            }
            "--solomon-name-copy-init" => {
                config.solomon_name_copy_init = true;
            }
            "--solomon-name-copy-repair" => {
                config.solomon_name_copy_repair = true;
            }
            "--solomon-name-copy-repair-preserve-body-output" => {
                config.solomon_name_copy_repair = true;
                config.solomon_name_copy_repair_preserve_body_output = true;
            }
            "--solomon-body-scaffold" => {
                config.solomon_body_scaffold = true;
            }
            "--solomon-body-opening-repair" => {
                config.solomon_body_opening_repair = true;
            }
            "--progress-interval-batches" => {
                config.progress_interval_batches =
                    parse_usize(args.next(), "--progress-interval-batches")?;
            }
            "--learning-rate" => {
                config.learning_rate = args.next().ok_or("--learning-rate requires N")?.parse()?;
                if config.learning_rate <= 0 {
                    return Err("--learning-rate must be positive".into());
                }
            }
            "--output-lr-shift" => {
                config.output_learning_rate_shift = parse_u8(args.next(), "--output-lr-shift")?;
            }
            "--mlp-lr-shift" => {
                config.mlp_learning_rate_shift = parse_u8(args.next(), "--mlp-lr-shift")?;
            }
            "--embed-lr-shift" | "--embedding-lr-shift" => {
                config.embedding_learning_rate_shift = parse_u8(args.next(), "--embed-lr-shift")?;
            }
            "--attention-lr-shift" => {
                config.attention_learning_rate_shift =
                    parse_u8(args.next(), "--attention-lr-shift")?;
            }
            "--attention-q-lr-shift" => {
                config.attention_q_learning_rate_shift =
                    parse_u8(args.next(), "--attention-q-lr-shift")?;
            }
            "--attention-qk-lr-shift" => {
                config.attention_qk_learning_rate_shift =
                    parse_u8(args.next(), "--attention-qk-lr-shift")?;
            }
            "--reject-loss-regression" => {
                config.reject_loss_regression = true;
            }
            "--adaptive-attention-shifts" => {
                config.adaptive_attention_shifts = true;
            }
            "--attention-vo-oracle" => {
                config.attention_vo_oracle = true;
            }
            "--max-text-tokens" => {
                config.max_text_tokens = parse_positive_usize(args.next(), "--max-text-tokens")?;
            }
            "--min-text-tokens" => {
                config.min_text_tokens = parse_usize(args.next(), "--min-text-tokens")?;
            }
            "--repeat-run-cap" => {
                config.repeat_run_cap = parse_usize(args.next(), "--repeat-run-cap")?;
            }
            "--no-repeat-ngram" => {
                config.no_repeat_ngram = parse_usize(args.next(), "--no-repeat-ngram")?;
            }
            "--conditioning-examples" => {
                let value = args
                    .next()
                    .ok_or("--conditioning-examples requires PATH or none")?;
                config.conditioning_examples = if value == "none" {
                    None
                } else {
                    Some(PathBuf::from(value))
                };
            }
            "--conditioning-boost-q8" => {
                config.conditioning_boost_q8 = args
                    .next()
                    .ok_or("--conditioning-boost-q8 requires N")?
                    .parse()?;
            }
            "--text-prior-examples" => {
                let value = args
                    .next()
                    .ok_or("--text-prior-examples requires PATH or none")?;
                config.text_prior_examples = if value == "none" {
                    None
                } else {
                    Some(PathBuf::from(value))
                };
            }
            "--text-prior-order" => {
                config.text_prior_order = parse_usize(args.next(), "--text-prior-order")?;
            }
            "--text-prior-min-order" => {
                config.text_prior_min_order = parse_usize(args.next(), "--text-prior-min-order")?;
            }
            "--text-prior-boost-q8" => {
                config.text_prior_boost_q8 = args
                    .next()
                    .ok_or("--text-prior-boost-q8 requires N")?
                    .parse()?;
            }
            "--text-prior-strict" => {
                config.text_prior_strict = true;
            }
            "--no-embedded-text-memory" => {
                config.use_embedded_text_memory = false;
            }
            "--embedded-text-lm-order" => {
                config.embedded_text_lm_order =
                    parse_usize(args.next(), "--embedded-text-lm-order")?;
            }
            "--text-chunk-boost-q8" => {
                config.text_chunk_boost_q8 = args
                    .next()
                    .ok_or("--text-chunk-boost-q8 requires N")?
                    .parse()?;
            }
            "--decode-logit-delta" => {
                config.decode_logit_delta = true;
            }
            "--prompt-name-opening-prior" => {
                config.prompt_name_opening_prior = true;
            }
            "--suppress-name-chunks-after-opening" => {
                config.suppress_name_chunks_after_opening = true;
            }
            "--top-k" => {
                config.top_k = parse_positive_usize(args.next(), "--top-k")?;
            }
            "--sample-seed" => {
                config.sample_seed = args.next().ok_or("--sample-seed requires N")?.parse()?;
            }
            "--eval-max-examples" => {
                let value = args
                    .next()
                    .ok_or("--eval-max-examples requires N or none")?;
                config.eval_max_examples = if value == "none" {
                    None
                } else {
                    Some(parse_positive_usize(Some(value), "--eval-max-examples")?)
                };
            }
            "--text-token-profile" => {
                config.text_token_profile = parse_text_token_profile(
                    &args
                        .next()
                        .ok_or("--text-token-profile requires char or chunked")?,
                )?;
            }
            "--position-policy" => {
                config.position_policy = parse_position_policy(
                    &args
                        .next()
                        .ok_or("--position-policy requires learned-absolute or nope")?,
                )?;
            }
            value => return Err(format!("unknown option: {value}").into()),
        }
    }
    if config.min_text_tokens > config.max_text_tokens {
        return Err("--min-text-tokens cannot exceed --max-text-tokens".into());
    }
    if config.target_token_min > config.target_token_max {
        return Err("--target-token-range MIN cannot exceed MAX".into());
    }
    Ok(config)
}

fn parse_position_policy(
    value: &str,
) -> Result<MiniTransformerPositionPolicy, Box<dyn std::error::Error>> {
    match value {
        "learned-absolute" | "learned_absolute" | "position-embedding" => {
            Ok(MiniTransformerPositionPolicy::LearnedAbsolute)
        }
        "nope" | "no-position" | "none" => Ok(MiniTransformerPositionPolicy::Nope),
        _ => Err(
            format!("unknown position policy: {value}; expected learned-absolute or nope").into(),
        ),
    }
}

fn parse_text_token_profile(value: &str) -> Result<TextTokenProfile, Box<dyn std::error::Error>> {
    match value {
        "char" => Ok(TextTokenProfile::Char),
        "chunked" => Ok(TextTokenProfile::Chunked),
        _ => Err(format!("unknown text token profile: {value}; expected char or chunked").into()),
    }
}

fn parse_target_phase(value: &str) -> Result<(u8, u8), Box<dyn std::error::Error>> {
    match value {
        "all" => Ok((u8::MIN, u8::MAX)),
        "special" => Ok((BOS, EOS)),
        "text-char" | "text" => Ok((TEXT_BASE + 32, TEXT_BASE + 126)),
        "text-chunk" | "chunk" => Ok((TEXT_CHUNK_BASE, u8::MAX)),
        "image" => Ok((IMAGE_BASE, IMAGE_BASE + IMAGE_BINS - 1)),
        _ => Err(format!(
            "unknown target phase: {value}; expected all, special, text-char, text-chunk, or image"
        )
        .into()),
    }
}

fn parse_target_segment(value: &str) -> Result<TargetSegment, Box<dyn std::error::Error>> {
    match value {
        "all" => Ok(TargetSegment::All),
        "generated-text" | "text" => Ok(TargetSegment::GeneratedText),
        "name-opening" | "opening-name" | "prompt-name" => Ok(TargetSegment::NameOpening),
        "name-opening-tail" | "opening-tail" | "prompt-name-tail" => {
            Ok(TargetSegment::NameOpeningTail)
        }
        "body-after-he" | "body" | "after-he" => Ok(TargetSegment::BodyAfterHe),
        "body-first-after-he" | "first-body-after-he" | "first-after-he" => {
            Ok(TargetSegment::BodyFirstAfterHe)
        }
        "body-first-after-opening"
        | "first-body-after-opening"
        | "body-start"
        | "first-after-opening" => Ok(TargetSegment::BodyFirstAfterOpening),
        "image" => Ok(TargetSegment::Image),
        _ => Err(format!(
            "unknown target segment: {value}; expected all, generated-text, name-opening, name-opening-tail, body-after-he, body-first-after-he, body-first-after-opening, or image"
        )
        .into()),
    }
}

fn parse_command(value: &str) -> Result<Command, Box<dyn std::error::Error>> {
    match value {
        "train" => Ok(Command::Train),
        "sample" => Ok(Command::Sample),
        "eval" => Ok(Command::Eval),
        _ => Err(format!("unknown command: {value}; expected train, sample, or eval").into()),
    }
}

fn parse_positive_usize(
    value: Option<String>,
    flag: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = parse_usize(value, flag)?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive").into());
    }
    Ok(parsed)
}

fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(value
        .ok_or_else(|| format!("{flag} requires N"))?
        .parse::<usize>()?)
}

fn parse_u8(value: Option<String>, flag: &str) -> Result<u8, Box<dyn std::error::Error>> {
    Ok(value
        .ok_or_else(|| format!("{flag} requires N"))?
        .parse::<u8>()?)
}

fn train_command(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = read_u8_tokens(&config.tokens_path)?;
    validate_corpus(&tokens)?;
    let train_config = train_config(&config);
    let initial = load_initial_transformer(&config, train_config.seq_len)?;
    let text_memory = load_text_memory(
        config.embed_text_memory_examples.as_ref(),
        &config.tokens_path,
        config.embed_text_memory_order,
    )?;
    let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
        &tokens,
        train_config,
        initial,
        config.progress_interval_batches,
        MiniTransformerTraceDetail::Summary,
        |progress| {
            println!(
                "{{\"schema\":\"nsrl.solomon_attention_progress.v1\",\"updates\":{},\"windows\":{},\"examined_windows\":{},\"accepted_batches\":{},\"rejected_batches\":{},\"model_hash\":\"0x{:016x}\"}}",
                progress.updates,
                progress.windows,
                progress.examined_windows,
                progress.accepted_batch_count,
                progress.rejected_batch_count,
                progress.model_hash,
            );
            Ok(())
        },
    )?;
    let mut transformer = run.model;
    if config.solomon_name_copy_repair {
        repair_solomon_name_copy_scaffold_with_options(
            &mut transformer,
            config.text_token_profile,
            config.solomon_name_copy_repair_preserve_body_output,
        )?;
    }
    if config.solomon_body_scaffold {
        apply_solomon_body_scaffold(&mut transformer, config.text_token_profile)?;
    }
    if config.solomon_body_opening_repair {
        let memory = text_memory
            .as_ref()
            .ok_or("--solomon-body-opening-repair requires --embed-text-memory-examples")?;
        repair_solomon_body_opening_scaffold(&mut transformer, config.text_token_profile, memory)?;
    }
    let model = SolomonAttentionModel {
        token_count: u64::try_from(tokens.len())?,
        token_hash: hash_bytes(&tokens),
        attention_kind: MiniTransformerAttentionKind::Base2Softmax,
        position_policy: config.position_policy,
        text_token_profile: config.text_token_profile,
        text_memory,
        transformer,
    };
    if let Some(parent) = config.model_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.model_out, model.try_to_bytes()?)?;
    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"token_count\":{},\"token_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\",\"inner_model_hash\":\"0x{:016x}\",\"attention_kind\":\"{}\",\"position_policy\":\"{}\",\"text_token_profile\":\"{}\",\"embedded_text_memory_order\":{},\"embedded_text_memory_examples\":{},\"adaptive_attention_shifts\":{},\"attention_vo_oracle\":{},\"zero_output_head_init\":{},\"solomon_name_copy_init\":{},\"solomon_name_copy_repair\":{},\"solomon_name_copy_repair_preserve_body_output\":{},\"solomon_body_scaffold\":{},\"solomon_body_opening_repair\":{},\"learning_rate\":{},\"output_lr_shift\":{},\"mlp_lr_shift\":{},\"embed_lr_shift\":{},\"attention_lr_shift\":{},\"attention_q_lr_shift\":{},\"attention_qk_lr_shift\":{},\"target_token_min\":{},\"target_token_max\":{},\"target_segment\":\"{}\",\"target_frequency_cap\":{},\"target_frequency_min_weight_q15\":{},\"argmax_margin_weight_q15\":{},\"reject_loss_regression\":{},\"seq_len\":{},\"stride\":{},\"window_offset\":{},\"windows\":{},\"examined_windows\":{},\"updates\":{},\"accepted_batches\":{},\"rejected_batches\":{},\"rollback_count\":{},\"rejected_windows\":{},\"final_accuracy_per_mille\":{},\"initial_probability_error_q15\":{},\"final_probability_error_q15\":{},\"probability_error_delta_i64\":{}}}",
        TRAIN_SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        model.token_count,
        model.token_hash,
        model.model_hash()?,
        model.transformer.model_hash(),
        model.attention_kind.as_str(),
        model.position_policy.as_str(),
        model.text_token_profile.as_str(),
        model
            .text_memory
            .as_ref()
            .map(|memory| memory.order)
            .unwrap_or(0),
        model
            .text_memory
            .as_ref()
            .map(|memory| memory.examples.len())
            .unwrap_or(0),
        config.adaptive_attention_shifts,
        config.attention_vo_oracle,
        config.zero_output_head_init,
        config.solomon_name_copy_init,
        config.solomon_name_copy_repair,
        config.solomon_name_copy_repair_preserve_body_output,
        config.solomon_body_scaffold,
        config.solomon_body_opening_repair,
        run.trace.config.learning_rate,
        run.trace.config.output_learning_rate_shift,
        run.trace.config.mlp_learning_rate_shift,
        run.trace.config.embedding_learning_rate_shift,
        run.trace.config.attention_learning_rate_shift,
        run.trace.config.attention_q_learning_rate_shift,
        run.trace.config.attention_qk_learning_rate_shift,
        run.trace.config.target_token_min,
        run.trace.config.target_token_max,
        config.target_segment.as_str(),
        run.trace.config.target_frequency_cap,
        run.trace.config.target_frequency_min_weight_q15,
        run.trace.config.argmax_margin_weight_q15,
        run.trace.config.reject_loss_regression,
        run.trace.config.seq_len,
        run.trace.config.stride,
        run.trace.config.window_offset,
        run.trace.windows,
        run.trace.examined_windows,
        run.trace.updates,
        run.trace.accepted_batch_count,
        run.trace.rejected_batch_count,
        run.trace.rollback_count,
        run.trace.rejected_window_count,
        run.trace.final_accuracy_per_mille,
        run.trace.initial_probability_error_q15,
        run.trace.final_probability_error_q15,
        run.trace.final_probability_error_q15 as i64
            - run.trace.initial_probability_error_q15 as i64,
    );
    Ok(())
}

fn sample_command(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let model = SolomonAttentionModel::from_bytes(&fs::read(&config.model_path)?)?;
    let conditioning = load_conditioning_match(
        config.conditioning_examples.as_ref(),
        &config.tokens_path,
        &config.prompt,
    )?;
    if !config.text_prefix.is_empty() && conditioning.is_some() {
        return Err("--text-prefix cannot be combined with prompt-conditioned examples".into());
    }
    let text_prior = load_text_prior(
        config.text_prior_examples.as_ref(),
        &config.tokens_path,
        config.text_prior_order,
        config.text_prior_min_order,
        model.text_token_profile,
    )?;
    let (
        text_prior_source,
        effective_text_prior,
        effective_text_prior_boost_q8,
        effective_text_prior_strict,
    ) = if let Some(text_prior) = text_prior.as_ref() {
        (
            "external",
            Some(text_prior.clone()),
            config.text_prior_boost_q8,
            config.text_prior_strict,
        )
    } else if config.use_embedded_text_memory
        && let Some(memory) = model.text_memory.as_ref()
    {
        (
            "embedded",
            Some(TextPrior::from(memory, model.text_token_profile)),
            if config.text_prior_boost_q8 > 0 {
                config.text_prior_boost_q8
            } else {
                DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8
            },
            true,
        )
    } else if config.embedded_text_lm_order > 0
        && let Some(memory) = model.text_memory.as_ref()
    {
        (
            "embedded_lm",
            Some(TextPrior::from_memory_for_prompt_with_order(
                memory,
                &config.prompt,
                config.embedded_text_lm_order,
                config.text_prior_min_order,
                true,
                model.text_token_profile,
            )),
            if config.text_prior_boost_q8 > 0 {
                config.text_prior_boost_q8
            } else {
                DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8
            },
            config.text_prior_strict,
        )
    } else {
        (
            "none",
            None,
            config.text_prior_boost_q8,
            config.text_prior_strict,
        )
    };
    let embedded_image_match = if conditioning.is_none() && config.use_embedded_text_memory {
        model
            .text_memory
            .as_ref()
            .and_then(|memory| memory.best_example_for_prompt(&config.prompt))
            .filter(|example| example.image_tokens.len() == SIGNATURE_BINS)
    } else {
        None
    };
    let sample = sample_model(
        &model,
        &config.prompt,
        SampleOptions {
            max_text_tokens: config.max_text_tokens,
            min_text_tokens: config.min_text_tokens,
            repeat_run_cap: config.repeat_run_cap,
            no_repeat_ngram: config.no_repeat_ngram,
            conditioning_boost_q8: config.conditioning_boost_q8,
            text_prior_boost_q8: effective_text_prior_boost_q8,
            text_prior_strict: effective_text_prior_strict,
            text_chunk_boost_q8: config.text_chunk_boost_q8,
            decode_logit_delta: config.decode_logit_delta,
            prompt_name_opening_prior: config.prompt_name_opening_prior,
            suppress_name_chunks_after_opening: config.suppress_name_chunks_after_opening,
            top_k: config.top_k,
            sample_seed: config.sample_seed,
        },
        conditioning.as_ref(),
        effective_text_prior.as_ref(),
        embedded_image_match
            .as_ref()
            .map(|example| example.image_tokens.as_slice()),
        &config.text_prefix,
    )?;
    fs::create_dir_all(&config.out_dir)?;
    let text_path = config.out_dir.join("text.txt");
    let ink_path = config.out_dir.join("image.ink16.u8");
    let pgm_path = config.out_dir.join("image.pgm");
    let token_path = config.out_dir.join("image.tokens.tsv");
    let trace_path = config.out_dir.join("sample.json");
    fs::write(&text_path, format!("{}\n", sample.text))?;
    fs::write(&ink_path, image_ink_bytes(&sample.image_bins))?;
    fs::write(&pgm_path, image_pgm_bytes(&sample.image_bins))?;
    fs::write(&token_path, image_token_tsv(&sample.image_bins))?;
    fs::write(
        &trace_path,
        format!(
            "{{\n  \"schema\":\"{}\",\n  \"model\":\"{}\",\n  \"model_hash\":\"0x{:016x}\",\n  \"inner_model_hash\":\"0x{:016x}\",\n  \"attention_kind\":\"{}\",\n  \"position_policy\":\"{}\",\n  \"text_token_profile\":\"{}\",\n  \"prompt\":\"{}\",\n  \"text_prefix\":\"{}\",\n  \"text_prefix_tokens\":{},\n  \"generated_text\":\"{}\",\n  \"generated_token_count\":{},\n  \"conditioning_primary_name\":\"{}\",\n  \"conditioning_prompt\":\"{}\",\n  \"conditioning_score\":{},\n  \"conditioning_text_tokens\":{},\n  \"conditioning_image_tokens\":{},\n  \"text_prior_source\":\"{}\",\n  \"text_prior_order\":{},\n  \"text_prior_min_order\":{},\n  \"text_prior_contexts\":{},\n  \"text_prior_prompt_starts\":{},\n  \"text_prior_selected_start_tokens\":{},\n  \"text_prior_boost_q8\":{},\n  \"text_prior_strict\":{},\n  \"text_chunk_boost_q8\":{},\n  \"image_prior_source\":\"{}\",\n  \"image_prior_primary_name\":\"{}\",\n  \"image_prior_prompt\":\"{}\",\n  \"image_prior_tokens\":{},\n  \"decode_logit_delta\":{},\n  \"prompt_name_opening_prior\":{},\n  \"suppress_name_chunks_after_opening\":{},\n  \"image_grid\":{},\n  \"image_bins\":{},\n  \"text_out\":\"{}\",\n  \"image_ink16_u8\":\"{}\",\n  \"image_pgm\":\"{}\"\n}}\n",
            SAMPLE_SCHEMA,
            json_escape(&config.model_path.display().to_string()),
            model.model_hash()?,
            model.transformer.model_hash(),
            model.attention_kind.as_str(),
            model.position_policy.as_str(),
            model.text_token_profile.as_str(),
            json_escape(&sample.prompt),
            json_escape(&config.text_prefix),
            encode_text_prefix_tokens(&config.text_prefix, model.text_token_profile).len(),
            json_escape(&sample.text),
            sample.generated_tokens.len(),
            json_escape(
                conditioning
                    .as_ref()
                    .map(|value| value.primary_name.as_str())
                    .unwrap_or("")
            ),
            json_escape(
                conditioning
                    .as_ref()
                    .map(|value| value.prompt.as_str())
                    .unwrap_or("")
            ),
            conditioning.as_ref().map(|value| value.score).unwrap_or(0),
            conditioning
                .as_ref()
                .map(|value| value.text_tokens.len())
                .unwrap_or(0),
            conditioning
                .as_ref()
                .map(|value| value.image_tokens.len())
                .unwrap_or(0),
            text_prior_source,
            effective_text_prior
                .as_ref()
                .map(|value| value.order)
                .unwrap_or(0),
            effective_text_prior
                .as_ref()
                .map(|value| value.min_order)
                .unwrap_or(0),
            effective_text_prior
                .as_ref()
                .map(|value| value.transitions.len())
                .unwrap_or(0),
            effective_text_prior
                .as_ref()
                .map(|value| value.prompt_starts.len())
                .unwrap_or(0),
            effective_text_prior
                .as_ref()
                .map(|value| value.start_tokens_for_prompt(&config.prompt).len())
                .unwrap_or(0),
            effective_text_prior_boost_q8,
            effective_text_prior_strict,
            config.text_chunk_boost_q8,
            if embedded_image_match.is_some() {
                "embedded"
            } else {
                "none"
            },
            embedded_image_match
                .as_ref()
                .map(|example| json_escape(&example.primary_name))
                .unwrap_or_default(),
            embedded_image_match
                .as_ref()
                .map(|example| json_escape(&example.prompt))
                .unwrap_or_default(),
            embedded_image_match
                .as_ref()
                .map(|example| example.image_tokens.len())
                .unwrap_or(0),
            config.decode_logit_delta,
            config.prompt_name_opening_prior,
            config.suppress_name_chunks_after_opening,
            SIGNATURE_GRID,
            IMAGE_BINS,
            json_escape(&text_path.display().to_string()),
            json_escape(&ink_path.display().to_string()),
            json_escape(&pgm_path.display().to_string()),
        ),
    )?;
    println!(
        "{{\"schema\":\"{}\",\"out_dir\":\"{}\",\"model_hash\":\"0x{:016x}\",\"prompt\":\"{}\",\"generated_text\":\"{}\"}}",
        SAMPLE_SCHEMA,
        json_escape(&config.out_dir.display().to_string()),
        model.model_hash()?,
        json_escape(&sample.prompt),
        json_escape(&sample.text),
    );
    Ok(())
}

fn eval_command(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let model = SolomonAttentionModel::from_bytes(&fs::read(&config.model_path)?)?;
    let tokens = read_u8_tokens(&config.tokens_path)?;
    validate_corpus(&tokens)?;
    let token_hash = hash_bytes(&tokens);
    if model.token_hash != token_hash {
        return Err(format!(
            "token hash mismatch: model has 0x{:016x}, corpus has 0x{token_hash:016x}",
            model.token_hash
        )
        .into());
    }

    let examples_path = config
        .conditioning_examples
        .as_ref()
        .ok_or("eval requires --conditioning-examples PATH")?;
    let mut examples = 0_usize;
    let mut skipped_examples = 0_usize;
    let mut total = EvalStats::default();
    let mut special = EvalStats::default();
    let mut prompt = EvalStats::default();
    let mut text = EvalStats::default();
    let mut image = EvalStats::default();
    let mut padded_context = PaddedContext::new(model.transformer.context_seq_len);

    for line in fs::read_to_string(examples_path)?
        .lines()
        .filter(|line| !line.is_empty())
    {
        if config
            .eval_max_examples
            .is_some_and(|limit| examples >= limit)
        {
            break;
        }
        let Some(offset) = json_usize_field(line, "token_offset") else {
            skipped_examples += 1;
            continue;
        };
        let Some(count) = json_usize_field(line, "token_count") else {
            skipped_examples += 1;
            continue;
        };
        let Some(end) = offset.checked_add(count) else {
            skipped_examples += 1;
            continue;
        };
        let Some(example_tokens) = tokens.get(offset..end) else {
            skipped_examples += 1;
            continue;
        };
        let Some(markers) = ExampleMarkers::from_tokens(example_tokens) else {
            skipped_examples += 1;
            continue;
        };
        examples += 1;
        for target_index in 1..example_tokens.len() {
            let phase = markers.phase_for_target(target_index);
            let padded_context_window = padded_context.window(
                &example_tokens[..target_index],
                model.transformer.context_seq_len,
            );
            let row = mini_transformer_next_token_row_with_attention_kind_position_policy(
                &model.transformer,
                padded_context_window,
                model.attention_kind,
                model.position_policy,
            )
            .or_else(|_| {
                next_token_row_with_context_backoff(&model, &example_tokens[..target_index])
            });
            let target = example_tokens[target_index];
            match row {
                Ok(row) => {
                    let diagnostics =
                        eval_token_diagnostics(&row, phase, model.text_token_profile, target);
                    let target_probability_q15 = u64::try_from(
                        i32::from(row.probabilities_q15[usize::from(target)]).max(0),
                    )?;
                    total.observe(target_probability_q15, diagnostics);
                    match phase {
                        EvalPhase::Special => special.observe(target_probability_q15, diagnostics),
                        EvalPhase::Prompt => prompt.observe(target_probability_q15, diagnostics),
                        EvalPhase::Text => text.observe(target_probability_q15, diagnostics),
                        EvalPhase::Image => image.observe(target_probability_q15, diagnostics),
                    }
                }
                Err(_) => {
                    total.observe_invalid();
                    match phase {
                        EvalPhase::Special => special.observe_invalid(),
                        EvalPhase::Prompt => prompt.observe_invalid(),
                        EvalPhase::Text => text.observe_invalid(),
                        EvalPhase::Image => image.observe_invalid(),
                    }
                }
            }
        }
    }

    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"model_hash\":\"0x{:016x}\",\"inner_model_hash\":\"0x{:016x}\",\"tokens\":\"{}\",\"token_count\":{},\"token_hash\":\"0x{:016x}\",\"examples\":\"{}\",\"eval_max_examples\":{},\"example_count\":{},\"skipped_examples\":{},\"attention_kind\":\"{}\",\"position_policy\":\"{}\",\"text_token_profile\":\"{}\",\"context_seq_len\":{},\"total\":{},\"special\":{},\"prompt\":{},\"text\":{},\"image\":{}}}",
        EVAL_SCHEMA,
        json_escape(&config.model_path.display().to_string()),
        model.model_hash()?,
        model.transformer.model_hash(),
        json_escape(&config.tokens_path.display().to_string()),
        tokens.len(),
        token_hash,
        json_escape(&examples_path.display().to_string()),
        config
            .eval_max_examples
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        examples,
        skipped_examples,
        model.attention_kind.as_str(),
        model.position_policy.as_str(),
        model.text_token_profile.as_str(),
        model.transformer.context_seq_len,
        total.to_json(),
        special.to_json(),
        prompt.to_json(),
        text.to_json(),
        image.to_json(),
    );
    Ok(())
}

fn train_config(config: &Config) -> MiniTransformerMlpTrainConfig {
    MiniTransformerMlpTrainConfig {
        epochs: config.epochs,
        seq_len: config.seq_len,
        stride: config.stride,
        window_offset: config.window_offset,
        max_windows: config.max_windows,
        batch_windows: config.batch_windows,
        target_token_min: config.target_token_min,
        target_token_max: config.target_token_max,
        target_segment: config
            .target_segment
            .to_train_segment(config.text_token_profile),
        target_frequency_cap: config.target_frequency_cap,
        target_frequency_min_weight_q15: config.target_frequency_min_weight_q15,
        argmax_margin_weight_q15: config.argmax_margin_weight_q15,
        tokenizer_id: ByteTokenizerId::Identity,
        attention_kind: MiniTransformerAttentionKind::Base2Softmax,
        position_policy: config.position_policy,
        learning_rate: config.learning_rate,
        output_learning_rate_shift: config.output_learning_rate_shift,
        mlp_learning_rate_shift: config.mlp_learning_rate_shift,
        embedding_learning_rate_shift: config.embedding_learning_rate_shift,
        attention_learning_rate_shift: config.attention_learning_rate_shift,
        attention_q_learning_rate_shift: config.attention_q_learning_rate_shift,
        attention_qk_learning_rate_shift: config.attention_qk_learning_rate_shift,
        adaptive_rule_shifts: false,
        adaptive_rule_interval_batches: 128,
        adaptive_attention_shifts: config.adaptive_attention_shifts,
        adaptive_holographic_shifts: false,
        attention_vo_error_feedback: false,
        attention_vo_oracle: config.attention_vo_oracle,
        reject_loss_regression: config.reject_loss_regression,
        batch_mode: MiniTransformerBatchMode::Serial,
        map_reduce_workers: 1,
    }
}

fn load_initial_transformer(
    config: &Config,
    seq_len: usize,
) -> Result<MiniTransformerMlpModel, Box<dyn std::error::Error>> {
    let Some(init_model) = config.init_model.as_ref() else {
        let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(seq_len);
        if config.zero_output_head_init {
            model.output_weights.fill(0);
        }
        if config.solomon_name_copy_init {
            apply_solomon_name_copy_init(&mut model, config.text_token_profile)?;
        }
        return Ok(model);
    };
    let model = SolomonAttentionModel::from_bytes(&fs::read(init_model)?)?;
    if model.transformer.context_seq_len != seq_len {
        return Err(format!(
            "--init-model context length {} does not match --seq-len {seq_len}",
            model.transformer.context_seq_len
        )
        .into());
    }
    if model.attention_kind != MiniTransformerAttentionKind::Base2Softmax {
        return Err("--init-model attention kind must be base2-softmax".into());
    }
    if model.position_policy != config.position_policy {
        return Err(format!(
            "--init-model position policy {} does not match --position-policy {}",
            model.position_policy.as_str(),
            config.position_policy.as_str()
        )
        .into());
    }
    if model.text_token_profile != config.text_token_profile {
        return Err(format!(
            "--init-model text token profile {} does not match --text-token-profile {}",
            model.text_token_profile.as_str(),
            config.text_token_profile.as_str()
        )
        .into());
    }
    let mut transformer = model.transformer;
    if config.zero_output_head_init {
        transformer.output_weights.fill(0);
    }
    if config.solomon_name_copy_init {
        apply_solomon_name_copy_init(&mut transformer, config.text_token_profile)?;
    }
    if config.solomon_body_scaffold {
        apply_solomon_body_scaffold(&mut transformer, config.text_token_profile)?;
    }
    Ok(transformer)
}

fn apply_solomon_name_copy_init(
    model: &mut MiniTransformerMlpModel,
    text_token_profile: TextTokenProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if text_token_profile != TextTokenProfile::Chunked {
        return Err("--solomon-name-copy-init requires --text-token-profile chunked".into());
    }
    if MINI_TRANSFORMER_HEADS == 0
        || !MINI_TRANSFORMER_D_MODEL.is_multiple_of(MINI_TRANSFORMER_HEADS)
    {
        return Err("invalid mini-transformer head layout for Solomon name copy init".into());
    }
    let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    if MINI_TRANSFORMER_HEADS < 2 || head_dim < 16 {
        return Err(
            "Solomon name copy init requires at least two heads with 16 dimensions each".into(),
        );
    }

    let embedding_len = VOCAB_SIZE
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy embedding shape overflow")?;
    if model.embeddings.len() != embedding_len {
        return Err("Solomon name copy init embedding shape mismatch".into());
    }

    model.position_embeddings.fill(0);
    model.q_weights.fill(0);
    model.k_weights.fill(0);
    model.v_weights.fill(0);
    model.o_weights.fill(0);
    model.up_weights.fill(0);
    model.gate_weights.fill(0);
    model.down_weights.fill(0);
    model.output_weights.fill(0);

    model.embeddings.fill(0);
    repair_solomon_name_copy_scaffold(model, text_token_profile)?;

    Ok(())
}

fn repair_solomon_name_copy_scaffold(
    model: &mut MiniTransformerMlpModel,
    text_token_profile: TextTokenProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    repair_solomon_name_copy_scaffold_with_options(model, text_token_profile, false)
}

fn repair_solomon_name_copy_scaffold_with_options(
    model: &mut MiniTransformerMlpModel,
    text_token_profile: TextTokenProfile,
    preserve_body_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if text_token_profile != TextTokenProfile::Chunked {
        return Err("--solomon-name-copy-repair requires --text-token-profile chunked".into());
    }
    if MINI_TRANSFORMER_HEADS == 0
        || !MINI_TRANSFORMER_D_MODEL.is_multiple_of(MINI_TRANSFORMER_HEADS)
    {
        return Err("invalid mini-transformer head layout for Solomon name copy repair".into());
    }
    let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    if MINI_TRANSFORMER_HEADS < 2 || head_dim < 16 {
        return Err(
            "Solomon name copy repair requires at least two heads with 16 dimensions each".into(),
        );
    }

    let embedding_len = VOCAB_SIZE
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy repair embedding shape overflow")?;
    if model.embeddings.len() != embedding_len {
        return Err("Solomon name copy repair embedding shape mismatch".into());
    }

    let prefix_token = usize::from(TEXT_CHUNK_BASE);
    let colon_token = usize::from(TEXT_CHUNK_BASE) + 1;
    let he_token = usize::from(TEXT_CHUNK_BASE) + 2;
    let code_dims = 2..9;
    let name_to_colon_dim = 14;
    let colon_to_he_dim = 15;
    let copied_code_base = head_dim;
    let name_start = usize::from(TEXT_CHUNK_BASE) + TEXT_CHUNK_NAME_START;
    let name_end = usize::from(TEXT_CHUNK_BASE) + TEXT_CHUNKS.len();

    for token in 0..VOCAB_SIZE {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        model.embeddings[row_start] = 0;
        model.embeddings[row_start + 1] = 0;
        model.embeddings[row_start + name_to_colon_dim] = 0;
        model.embeddings[row_start + colon_to_he_dim] = 0;
        for dim in code_dims.clone() {
            model.embeddings[row_start + dim] = 0;
            model.embeddings[row_start + copied_code_base + dim] = 0;
        }
    }
    for position in 0..model.position_embeddings.len() / MINI_TRANSFORMER_D_MODEL {
        let row_start = position * MINI_TRANSFORMER_D_MODEL;
        model.position_embeddings[row_start] = 0;
        model.position_embeddings[row_start + 1] = 0;
        model.position_embeddings[row_start + name_to_colon_dim] = 0;
        model.position_embeddings[row_start + colon_to_he_dim] = 0;
        for dim in code_dims.clone() {
            model.position_embeddings[row_start + dim] = 0;
            model.position_embeddings[row_start + copied_code_base + dim] = 0;
        }
    }
    for dim in 0..head_dim {
        clear_square_row(&mut model.q_weights, dim)?;
        clear_square_row(&mut model.k_weights, dim)?;
    }
    for dim in code_dims.clone() {
        clear_square_row(&mut model.v_weights, dim)?;
        clear_square_row(&mut model.o_weights, copied_code_base + dim)?;
    }
    if preserve_body_output {
        clear_output_row(&mut model.output_weights, prefix_token)?;
    } else {
        for token in 0..VOCAB_SIZE {
            let row_start = token * MINI_TRANSFORMER_D_MODEL;
            for dim in 0..head_dim {
                model.output_weights[row_start + dim] = 0;
            }
            for dim in code_dims.clone() {
                model.output_weights[row_start + copied_code_base + dim] = 0;
            }
        }
    }
    for token in name_start..name_end {
        clear_output_row(&mut model.output_weights, token)?;
    }
    clear_output_row(&mut model.output_weights, colon_token)?;
    clear_output_row(&mut model.output_weights, he_token)?;

    model.embeddings[prefix_token * MINI_TRANSFORMER_D_MODEL] = 4096;
    model.embeddings[colon_token * MINI_TRANSFORMER_D_MODEL + colon_to_he_dim] = 4096;
    model.output_weights[colon_token * MINI_TRANSFORMER_D_MODEL + name_to_colon_dim] = 127;
    model.output_weights[he_token * MINI_TRANSFORMER_D_MODEL + colon_to_he_dim] = 127;

    for token in name_start..name_end {
        let name_index = token - name_start;
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        model.embeddings[row_start + 1] = 4096;
        model.embeddings[row_start + name_to_colon_dim] = 4096;
        for (bit_index, dim) in code_dims.clone().enumerate() {
            let code = solomon_name_copy_code_q15(name_index, bit_index);
            let copied_dim = copied_code_base + dim;
            model.embeddings[row_start + dim] = code;
            model.output_weights[token * MINI_TRANSFORMER_D_MODEL + copied_dim] =
                if code >= 0 { 127 } else { -127 };
        }
    }

    set_square_i8(&mut model.q_weights, 0, 0, 8)?;
    set_square_i8(&mut model.k_weights, 0, 1, 8)?;
    for dim in code_dims {
        set_square_i8(&mut model.v_weights, dim, dim, 1)?;
        set_square_i8(&mut model.o_weights, copied_code_base + dim, dim, 1)?;
    }

    Ok(())
}

fn clear_output_row(weights: &mut [i8], token: usize) -> Result<(), Box<dyn std::error::Error>> {
    let row_start = token
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy output row overflow")?;
    let row_end = row_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy output row overflow")?;
    let row = weights
        .get_mut(row_start..row_end)
        .ok_or("Solomon name copy output shape mismatch")?;
    row.fill(0);
    Ok(())
}

fn clear_square_row(
    weights: &mut [i8],
    output_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let row_start = output_dim
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy weight row overflow")?;
    let row_end = row_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon name copy weight row overflow")?;
    let row = weights
        .get_mut(row_start..row_end)
        .ok_or("Solomon name copy weight shape mismatch")?;
    row.fill(0);
    Ok(())
}

fn set_square_i8(
    weights: &mut [i8],
    output_dim: usize,
    input_dim: usize,
    value: i8,
) -> Result<(), Box<dyn std::error::Error>> {
    let index = output_dim
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .and_then(|start| start.checked_add(input_dim))
        .ok_or("Solomon name copy weight index overflow")?;
    let slot = weights
        .get_mut(index)
        .ok_or("Solomon name copy weight shape mismatch")?;
    *slot = value;
    Ok(())
}

fn solomon_name_copy_code_q15(name_index: usize, bit_index: usize) -> i16 {
    if ((name_index + 1) >> bit_index) & 1 == 1 {
        4096
    } else {
        -4096
    }
}

fn apply_solomon_body_scaffold(
    model: &mut MiniTransformerMlpModel,
    text_token_profile: TextTokenProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if text_token_profile != TextTokenProfile::Chunked {
        return Err("--solomon-body-scaffold requires --text-token-profile chunked".into());
    }
    let body_dims = 25..MINI_TRANSFORMER_D_MODEL;
    if body_dims.is_empty() || body_dims.start <= MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS
    {
        return Err("Solomon body scaffold requires a free high model dimension".into());
    }
    let embedding_len = VOCAB_SIZE
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon body scaffold embedding shape overflow")?;
    if model.embeddings.len() != embedding_len {
        return Err("Solomon body scaffold embedding shape mismatch".into());
    }
    if model.output_weights.len() != embedding_len {
        return Err("Solomon body scaffold output shape mismatch".into());
    }

    for token in 0..VOCAB_SIZE {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        for dim in body_dims.clone() {
            model.embeddings[row_start + dim] = 0;
            model.output_weights[row_start + dim] = 0;
        }
    }

    for (transition_index, (previous, next)) in
        solomon_body_scaffold_transitions()?.into_iter().enumerate()
    {
        let previous_row = usize::from(previous) * MINI_TRANSFORMER_D_MODEL;
        let next_row = usize::from(next) * MINI_TRANSFORMER_D_MODEL;
        for (bit_index, dim) in body_dims.clone().enumerate() {
            let code = solomon_name_copy_code_q15(transition_index, bit_index);
            model.embeddings[previous_row + dim] = code;
            model.output_weights[next_row + dim] = if code >= 0 { 127 } else { -127 };
        }
    }

    Ok(())
}

fn repair_solomon_body_opening_scaffold(
    model: &mut MiniTransformerMlpModel,
    text_token_profile: TextTokenProfile,
    memory: &TextMemory,
) -> Result<(), Box<dyn std::error::Error>> {
    if text_token_profile != TextTokenProfile::Chunked {
        return Err("--solomon-body-opening-repair requires --text-token-profile chunked".into());
    }
    if MINI_TRANSFORMER_HEADS == 0
        || !MINI_TRANSFORMER_D_MODEL.is_multiple_of(MINI_TRANSFORMER_HEADS)
    {
        return Err("invalid mini-transformer head layout for Solomon body opening repair".into());
    }
    let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    if head_dim < 16 {
        return Err("Solomon body opening repair requires a 16-dimensional first head".into());
    }

    let embedding_len = VOCAB_SIZE
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or("Solomon body opening repair embedding shape overflow")?;
    if model.embeddings.len() != embedding_len {
        return Err("Solomon body opening repair embedding shape mismatch".into());
    }
    if model.output_weights.len() != embedding_len {
        return Err("Solomon body opening repair output shape mismatch".into());
    }

    let assignments = solomon_body_opening_assignments(memory, text_token_profile)?;
    let mut class_tokens = Vec::<u8>::new();
    let mut name_classes = Vec::<(usize, usize)>::new();
    for (name_index, token) in assignments {
        let class_index = if let Some(index) = class_tokens.iter().position(|&seen| seen == token) {
            index
        } else {
            if class_tokens.len() >= 8 {
                return Err(
                    "Solomon body opening repair supports at most eight opening classes".into(),
                );
            }
            class_tokens.push(token);
            class_tokens.len() - 1
        };
        name_classes.push((name_index, class_index));
    }

    let colon_token = usize::from(TEXT_CHUNK_BASE) + 1;
    let query_input_dim = head_dim;
    let key_input_dim = head_dim + 1;
    let code_dims = head_dim + 9..head_dim + 12;
    let copied_code_dims = [head_dim + 12, head_dim + 13, head_dim + 14, head_dim + 15];
    let scaffold_output_tokens = solomon_body_scaffold_transitions()?
        .into_iter()
        .map(|(_, next)| next)
        .collect::<Vec<_>>();

    for token in 0..VOCAB_SIZE {
        let row_start = token * MINI_TRANSFORMER_D_MODEL;
        model.embeddings[row_start + query_input_dim] = 0;
        model.embeddings[row_start + key_input_dim] = 0;
        model.embeddings[row_start + copied_code_dims[0]] = 0;
        model.output_weights[row_start + copied_code_dims[0]] = 0;
    }
    for position in 0..model.position_embeddings.len() / MINI_TRANSFORMER_D_MODEL {
        let row_start = position * MINI_TRANSFORMER_D_MODEL;
        model.position_embeddings[row_start + query_input_dim] = 0;
        model.position_embeddings[row_start + key_input_dim] = 0;
        model.position_embeddings[row_start + copied_code_dims[0]] = 0;
    }

    clear_square_row(&mut model.q_weights, query_input_dim)?;
    clear_square_row(&mut model.k_weights, query_input_dim)?;
    for dim in code_dims.clone() {
        clear_square_row(&mut model.v_weights, dim)?;
        set_square_i8(&mut model.v_weights, dim, dim, 1)?;
    }
    for (index, &output_dim) in copied_code_dims.iter().enumerate() {
        let input_dim = code_dims.start + index % code_dims.len();
        clear_square_row(&mut model.o_weights, output_dim)?;
        set_square_i8(&mut model.o_weights, output_dim, input_dim, 1)?;
    }
    set_square_i8(&mut model.q_weights, query_input_dim, query_input_dim, 8)?;
    set_square_i8(&mut model.k_weights, query_input_dim, key_input_dim, 8)?;

    model.embeddings[colon_token * MINI_TRANSFORMER_D_MODEL + query_input_dim] = 4096;
    for &(name_index, class_index) in &name_classes {
        let name_token = usize::from(TEXT_CHUNK_BASE) + TEXT_CHUNK_NAME_START + name_index;
        let row_start = name_token * MINI_TRANSFORMER_D_MODEL;
        model.embeddings[row_start + key_input_dim] = 4096;
        for (bit_index, dim) in code_dims.clone().enumerate() {
            model.embeddings[row_start + dim] =
                solomon_body_opening_class_code_q15(class_index, bit_index);
        }
    }
    for (class_index, &token) in class_tokens.iter().enumerate() {
        let preserve_scaffold_output = scaffold_output_tokens.contains(&token);
        let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        for (index, &dim) in copied_code_dims.iter().enumerate() {
            let bit_index = index % code_dims.len();
            let value = if solomon_body_opening_class_code_q15(class_index, bit_index) >= 0 {
                127
            } else {
                -127
            };
            if preserve_scaffold_output {
                model.output_weights[row_start + dim] =
                    model.output_weights[row_start + dim].saturating_add(value / 2);
            } else {
                model.output_weights[row_start + dim] = value;
            }
        }
    }

    Ok(())
}

fn solomon_body_opening_assignments(
    memory: &TextMemory,
    text_token_profile: TextTokenProfile,
) -> Result<Vec<(usize, u8)>, Box<dyn std::error::Error>> {
    let mut assignments = vec![None; TEXT_CHUNKS.len().saturating_sub(TEXT_CHUNK_NAME_START)];
    for example in &memory.examples {
        let Some(name_index) = solomon_name_chunk_index(&example.primary_name) else {
            continue;
        };
        let text_tokens = sanitize_source_text_tokens(&example.text_tokens, text_token_profile);
        let prefix = text_prefix_through_colon(&text_tokens);
        let Some(&token) = text_tokens.get(prefix.len()) else {
            continue;
        };
        if !is_text_token(token) {
            continue;
        }
        match assignments[name_index] {
            Some(previous) if previous != token => {
                return Err(format!(
                    "conflicting body opening token for {}",
                    example.primary_name
                )
                .into());
            }
            _ => assignments[name_index] = Some(token),
        }
    }
    assignments
        .into_iter()
        .enumerate()
        .map(|(name_index, token)| {
            token.map(|token| (name_index, token)).ok_or_else(|| {
                format!(
                    "missing body opening token for {}",
                    TEXT_CHUNKS[TEXT_CHUNK_NAME_START + name_index]
                )
                .into()
            })
        })
        .collect()
}

fn solomon_name_chunk_index(name: &str) -> Option<usize> {
    let name_key = normalize_key(name);
    TEXT_CHUNKS
        .get(TEXT_CHUNK_NAME_START..)?
        .iter()
        .position(|candidate| normalize_key(candidate) == name_key)
}

fn solomon_body_opening_class_code_q15(class_index: usize, bit_index: usize) -> i16 {
    solomon_name_copy_code_q15(class_index, bit_index)
}

fn solomon_body_scaffold_transitions() -> Result<Vec<(u8, u8)>, Box<dyn std::error::Error>> {
    let body_tokens = encode_text_prefix_tokens(
        "He is of the Goetia and teacheth with his ART in LINE.",
        TextTokenProfile::Chunked,
    );
    let he_token = TEXT_CHUNK_BASE + 2;
    if body_tokens.first().copied() != Some(he_token) {
        return Err("Solomon body scaffold phrase must start with He chunk".into());
    }
    let mut transitions = Vec::with_capacity(body_tokens.len());
    for pair in body_tokens.windows(2) {
        transitions.push((pair[0], pair[1]));
    }
    transitions.push((
        *body_tokens
            .last()
            .ok_or("Solomon body scaffold phrase must not be empty")?,
        IMAGE,
    ));
    Ok(transitions)
}

fn read_u8_tokens(path: &PathBuf) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Ok(fs::read(path)?)
}

fn validate_corpus(tokens: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    if tokens.is_empty() {
        return Err("cannot train attention model from an empty corpus".into());
    }
    for (index, &token) in tokens.iter().enumerate() {
        if usize::from(token) >= VOCAB_SIZE {
            return Err(
                format!("token {token} at offset {index} is outside NSRLLMM1 vocab").into(),
            );
        }
    }
    Ok(())
}

fn load_conditioning_match(
    examples_path: Option<&PathBuf>,
    tokens_path: &PathBuf,
    prompt: &str,
) -> Result<Option<ConditioningMatch>, Box<dyn std::error::Error>> {
    let Some(examples_path) = examples_path else {
        return Ok(None);
    };
    if !examples_path.exists() {
        return Ok(None);
    }
    let tokens = read_u8_tokens(tokens_path)?;
    let normalized_prompt = normalize_text(prompt);
    let mut best: Option<ConditioningMatch> = None;

    for line in fs::read_to_string(examples_path)?
        .lines()
        .filter(|line| !line.is_empty())
    {
        let primary_name = json_string_field(line, "primary_name").unwrap_or_default();
        let example_prompt = json_string_field(line, "prompt").unwrap_or_default();
        let Some(offset) = json_usize_field(line, "token_offset") else {
            continue;
        };
        let Some(count) = json_usize_field(line, "token_count") else {
            continue;
        };
        let Some(end) = offset.checked_add(count) else {
            continue;
        };
        let Some(example_tokens) = tokens.get(offset..end) else {
            continue;
        };
        let Some((text_tokens, image_tokens)) = extract_example_segments(example_tokens) else {
            continue;
        };
        let score = conditioning_score(&normalized_prompt, &example_prompt, &primary_name);
        if score == 0 {
            continue;
        }
        let candidate = ConditioningMatch {
            primary_name,
            prompt: example_prompt,
            score,
            text_tokens,
            image_tokens,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.score > current.score)
        {
            best = Some(candidate);
        }
    }

    Ok(best)
}

fn extract_example_segments(tokens: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let text_marker = tokens.iter().position(|&token| token == TEXT)?;
    let image_marker = tokens.iter().position(|&token| token == IMAGE)?;
    if image_marker <= text_marker {
        return None;
    }
    let eos_marker = tokens
        .iter()
        .enumerate()
        .skip(image_marker + 1)
        .find_map(|(index, &token)| if token == EOS { Some(index) } else { None })?;
    let text_tokens = tokens[text_marker + 1..image_marker]
        .iter()
        .copied()
        .filter(|&token| is_text_token(token))
        .collect();
    let image_tokens: Vec<u8> = tokens[image_marker + 1..eos_marker]
        .iter()
        .copied()
        .filter(|&token| is_image_token(token))
        .collect();
    if image_tokens.len() != SIGNATURE_BINS {
        return None;
    }
    Some((text_tokens, image_tokens))
}

fn load_text_prior(
    examples_path: Option<&PathBuf>,
    tokens_path: &PathBuf,
    order: usize,
    min_order: usize,
    text_token_profile: TextTokenProfile,
) -> Result<Option<TextPrior>, Box<dyn std::error::Error>> {
    Ok(
        load_text_memory(examples_path, tokens_path, order)?.map(|memory| {
            TextPrior::from_memory_with_order(&memory, order, min_order, true, text_token_profile)
        }),
    )
}

fn load_text_memory(
    examples_path: Option<&PathBuf>,
    tokens_path: &PathBuf,
    order: usize,
) -> Result<Option<TextMemory>, Box<dyn std::error::Error>> {
    let Some(examples_path) = examples_path else {
        return Ok(None);
    };
    if !examples_path.exists() {
        return Ok(None);
    }
    let tokens = read_u8_tokens(tokens_path)?;
    let mut memory = TextMemory {
        order,
        examples: Vec::new(),
    };
    for line in fs::read_to_string(examples_path)?
        .lines()
        .filter(|line| !line.is_empty())
    {
        let primary_name = json_string_field(line, "primary_name").unwrap_or_default();
        let example_prompt = json_string_field(line, "prompt").unwrap_or_default();
        let Some(offset) = json_usize_field(line, "token_offset") else {
            continue;
        };
        let Some(count) = json_usize_field(line, "token_count") else {
            continue;
        };
        let Some(end) = offset.checked_add(count) else {
            continue;
        };
        let Some(example_tokens) = tokens.get(offset..end) else {
            continue;
        };
        let Some((text_tokens, image_tokens)) = extract_example_segments(example_tokens) else {
            continue;
        };
        memory.examples.push(TextMemoryExample {
            primary_name,
            prompt: example_prompt,
            text_tokens,
            image_tokens,
        });
    }
    Ok(Some(memory))
}

fn add_text_prior_sequence(prior: &mut TextPrior, text_tokens: &[u8]) {
    if prior.start_tokens.is_empty() {
        prior.start_tokens.extend_from_slice(text_tokens);
    } else {
        let common_len = prior
            .start_tokens
            .iter()
            .zip(text_tokens.iter())
            .take_while(|(left, right)| left == right)
            .count();
        prior.start_tokens.truncate(common_len);
    }
    if let Some(&first_token) = text_tokens.first() {
        let counts = prior
            .transitions
            .entry(vec![TEXT])
            .or_insert([0_u32; VOCAB_LEN]);
        counts[usize::from(first_token)] = counts[usize::from(first_token)].saturating_add(1);
    }
    for index in 0..=text_tokens.len() {
        let target = if index == text_tokens.len() {
            IMAGE
        } else {
            text_tokens[index]
        };
        let max_order = prior.order.min(index);
        for order in 0..=max_order {
            let context = text_tokens[index - order..index].to_vec();
            let counts = prior
                .transitions
                .entry(context)
                .or_insert([0_u32; VOCAB_LEN]);
            counts[usize::from(target)] = counts[usize::from(target)].saturating_add(1);
        }
    }
}

fn add_text_prior_prompt_start(
    prior: &mut TextPrior,
    primary_name: &str,
    prompt: &str,
    text_tokens: &[u8],
) {
    let tokens = text_prefix_through_colon(text_tokens);
    if tokens.is_empty() {
        return;
    }
    let primary_key = normalize_key(primary_name);
    let prompt_key = normalize_key(prompt);
    let mut keys = Vec::new();
    if !prompt_key.is_empty() && !is_generic_prompt_key(&prompt_key) {
        keys.push(prompt_key);
    }
    if !primary_key.is_empty() {
        keys.push(primary_key.clone());
        keys.push(normalize_key(&format!("seal of {primary_name}")));
    }
    keys.sort();
    keys.dedup();
    for key in keys {
        if prior
            .prompt_starts
            .iter()
            .any(|start| start.prompt_key == key && start.primary_key == primary_key)
        {
            continue;
        }
        prior.prompt_starts.push(PromptTextStart {
            prompt_key: key,
            primary_key: primary_key.clone(),
            tokens: tokens.clone(),
        });
    }
}

fn text_prefix_through_colon(text_tokens: &[u8]) -> Vec<u8> {
    let mut end = None;
    for (index, &token) in text_tokens.iter().enumerate() {
        if token == TEXT_BASE + b':' || text_chunk_contains(token, ':') {
            end = Some(index + 1);
            break;
        }
    }
    let Some(mut end) = end else {
        return Vec::new();
    };
    if text_tokens.get(end).copied() == Some(TEXT_BASE + b' ') {
        end += 1;
    }
    text_tokens[..end].to_vec()
}

fn text_chunk_contains(token: u8, needle: char) -> bool {
    if !is_text_chunk_token(token) {
        return false;
    }
    let chunk_index = usize::from(token.saturating_sub(TEXT_CHUNK_BASE));
    TEXT_CHUNKS
        .get(chunk_index)
        .is_some_and(|chunk| chunk.contains(needle))
}

fn is_generic_prompt_key(prompt_key: &str) -> bool {
    prompt_key == "king solomon seal"
}

impl TextPrior {
    fn from(memory: &TextMemory, text_token_profile: TextTokenProfile) -> Self {
        Self::from_memory_with_order(memory, memory.order, 0, true, text_token_profile)
    }

    fn from_memory_with_order(
        memory: &TextMemory,
        order: usize,
        min_order: usize,
        include_prompt_starts: bool,
        text_token_profile: TextTokenProfile,
    ) -> Self {
        Self::from_examples_with_order(
            memory.examples.iter(),
            order,
            min_order,
            include_prompt_starts,
            text_token_profile,
        )
    }

    fn from_memory_for_prompt_with_order(
        memory: &TextMemory,
        prompt: &str,
        order: usize,
        min_order: usize,
        include_prompt_starts: bool,
        text_token_profile: TextTokenProfile,
    ) -> Self {
        let normalized_prompt = normalize_text(prompt);
        let mut matching = memory
            .examples
            .iter()
            .filter(|example| {
                text_memory_prompt_scope_score(
                    &normalized_prompt,
                    &example.prompt,
                    &example.primary_name,
                ) > 0
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            matching = memory.examples.iter().collect();
        }
        Self::from_examples_with_order(
            matching.iter().copied(),
            order,
            min_order,
            include_prompt_starts,
            text_token_profile,
        )
    }

    fn from_examples_with_order<'a, I>(
        examples: I,
        order: usize,
        min_order: usize,
        include_prompt_starts: bool,
        text_token_profile: TextTokenProfile,
    ) -> Self
    where
        I: IntoIterator<Item = &'a TextMemoryExample>,
    {
        let mut prior = Self {
            order,
            min_order: min_order.min(order),
            start_tokens: Vec::new(),
            prompt_starts: Vec::new(),
            transitions: HashMap::new(),
        };
        for example in examples {
            let text_tokens = sanitize_source_text_tokens(&example.text_tokens, text_token_profile);
            add_text_prior_sequence(&mut prior, &text_tokens);
            if include_prompt_starts {
                add_text_prior_prompt_start(
                    &mut prior,
                    &example.primary_name,
                    &example.prompt,
                    &text_tokens,
                );
            }
        }
        prior
    }

    fn start_tokens_for_prompt<'a>(&'a self, prompt: &str) -> &'a [u8] {
        let prompt_key = normalize_key(prompt);
        let mut best: Option<(&PromptTextStart, usize)> = None;
        for start in &self.prompt_starts {
            let score = prompt_start_score(&prompt_key, start);
            if score == 0 {
                continue;
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((start, score));
            }
        }
        best.map(|(start, _)| start.tokens.as_slice())
            .unwrap_or(&self.start_tokens)
    }
}

impl TextMemory {
    fn best_example_for_prompt(&self, prompt: &str) -> Option<&TextMemoryExample> {
        let prompt_key = normalize_key(prompt);
        let mut best: Option<(&TextMemoryExample, usize)> = None;
        for example in &self.examples {
            let score =
                text_memory_prompt_scope_score(&prompt_key, &example.prompt, &example.primary_name);
            if score == 0 {
                continue;
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((example, score));
            }
        }
        best.map(|(example, _)| example)
            .or_else(|| self.examples.first())
    }
}

fn text_memory_prompt_scope_score(prompt: &str, example_prompt: &str, primary_name: &str) -> usize {
    let prompt_key = normalize_key(prompt);
    let example_prompt_key = normalize_key(example_prompt);
    if prompt_key == example_prompt_key {
        return 1_000_000;
    }
    if is_generic_prompt_key(&prompt_key) {
        return 1;
    }
    let primary_key = normalize_key(primary_name);
    if !primary_key.is_empty() && prompt_contains_phrase(&prompt_key, &primary_key) {
        return 100_000;
    }
    0
}

fn prompt_start_score(prompt_key: &str, start: &PromptTextStart) -> usize {
    if prompt_key == start.prompt_key {
        return 1_000_000;
    }
    if !start.primary_key.is_empty() && prompt_contains_phrase(prompt_key, &start.primary_key) {
        return 100_000;
    }
    0
}

fn prompt_contains_phrase(prompt_key: &str, phrase: &str) -> bool {
    prompt_key == phrase
        || prompt_key.split_whitespace().any(|word| word == phrase)
        || prompt_key.contains(&format!(" {phrase} "))
        || prompt_key.strip_prefix(&format!("{phrase} ")).is_some()
        || prompt_key.strip_suffix(&format!(" {phrase}")).is_some()
}

impl ExampleMarkers {
    fn from_tokens(tokens: &[u8]) -> Option<Self> {
        let prompt = tokens.iter().position(|&token| token == PROMPT)?;
        let text = tokens.iter().position(|&token| token == TEXT)?;
        let image = tokens.iter().position(|&token| token == IMAGE)?;
        let eos = tokens
            .iter()
            .enumerate()
            .skip(image + 1)
            .find_map(|(index, &token)| if token == EOS { Some(index) } else { None })?;
        if prompt != 1 || text <= prompt || image <= text || eos <= image {
            return None;
        }
        Some(Self {
            prompt,
            text,
            image,
            eos,
        })
    }

    fn phase_for_target(self, target_index: usize) -> EvalPhase {
        if target_index > self.prompt && target_index < self.text {
            EvalPhase::Prompt
        } else if target_index > self.text && target_index < self.image {
            EvalPhase::Text
        } else if target_index > self.image && target_index < self.eos {
            EvalPhase::Image
        } else {
            EvalPhase::Special
        }
    }
}

impl EvalStats {
    fn observe(&mut self, target_probability_q15: u64, diagnostics: EvalTokenDiagnostics) {
        let first_valid = self.targets == self.invalid_contexts;
        self.targets += 1;
        if diagnostics.target_rank == 1 {
            self.correct += 1;
        }
        if diagnostics.target_rank <= 5 {
            self.top5_correct += 1;
        }
        if diagnostics.target_rank <= 10 {
            self.top10_correct += 1;
        }
        self.target_rank_sum = self
            .target_rank_sum
            .saturating_add(u64::try_from(diagnostics.target_rank).unwrap_or(u64::MAX));
        self.target_margin_q8_sum = self
            .target_margin_q8_sum
            .saturating_add(i64::from(diagnostics.target_margin_q8));
        if first_valid || diagnostics.target_margin_q8 < self.target_margin_q8_min {
            self.target_margin_q8_min = diagnostics.target_margin_q8;
        }
        self.probability_error_q15 = self
            .probability_error_q15
            .saturating_add(32_767_u64.saturating_sub(target_probability_q15.min(32_767)));
    }

    fn observe_invalid(&mut self) {
        self.targets += 1;
        self.invalid_contexts += 1;
        self.probability_error_q15 = self.probability_error_q15.saturating_add(32_767);
    }

    fn accuracy_per_mille(self) -> usize {
        if self.targets == 0 {
            return 0;
        }
        self.correct.saturating_mul(1000) / self.targets
    }

    fn mean_probability_error_q15(self) -> u64 {
        if self.targets == 0 {
            return 0;
        }
        self.probability_error_q15 / u64::try_from(self.targets).unwrap_or(1)
    }

    fn valid_targets(self) -> usize {
        self.targets.saturating_sub(self.invalid_contexts)
    }

    fn top5_accuracy_per_mille(self) -> usize {
        let valid = self.valid_targets();
        if valid == 0 {
            return 0;
        }
        self.top5_correct.saturating_mul(1000) / valid
    }

    fn top10_accuracy_per_mille(self) -> usize {
        let valid = self.valid_targets();
        if valid == 0 {
            return 0;
        }
        self.top10_correct.saturating_mul(1000) / valid
    }

    fn mean_target_rank_per_mille(self) -> u64 {
        let valid = self.valid_targets();
        if valid == 0 {
            return 0;
        }
        self.target_rank_sum.saturating_mul(1000) / u64::try_from(valid).unwrap_or(1)
    }

    fn mean_target_margin_q8(self) -> i64 {
        let valid = self.valid_targets();
        if valid == 0 {
            return 0;
        }
        self.target_margin_q8_sum / i64::try_from(valid).unwrap_or(1)
    }

    fn to_json(self) -> String {
        format!(
            "{{\"targets\":{},\"correct\":{},\"invalid_contexts\":{},\"accuracy_per_mille\":{},\"top5_accuracy_per_mille\":{},\"top10_accuracy_per_mille\":{},\"mean_target_rank_per_mille\":{},\"mean_target_margin_q8\":{},\"min_target_margin_q8\":{},\"probability_error_q15\":{},\"mean_probability_error_q15\":{}}}",
            self.targets,
            self.correct,
            self.invalid_contexts,
            self.accuracy_per_mille(),
            self.top5_accuracy_per_mille(),
            self.top10_accuracy_per_mille(),
            self.mean_target_rank_per_mille(),
            self.mean_target_margin_q8(),
            self.target_margin_q8_min,
            self.probability_error_q15,
            self.mean_probability_error_q15(),
        )
    }
}

fn eval_token_diagnostics(
    row: &MiniTransformerNextTokenRow,
    phase: EvalPhase,
    text_token_profile: TextTokenProfile,
    target: u8,
) -> EvalTokenDiagnostics {
    let mut candidates = eval_allowed_tokens(phase, text_token_profile)
        .into_iter()
        .map(|token| Candidate {
            token,
            logit_q8: row.logits_q8[usize::from(token)],
            probability_q15: i32::from(row.probabilities_q15[usize::from(token)]).max(0),
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .logit_q8
            .cmp(&left.logit_q8)
            .then_with(|| right.probability_q15.cmp(&left.probability_q15))
            .then_with(|| left.token.cmp(&right.token))
    });
    let predicted = candidates
        .first()
        .map(|candidate| candidate.token)
        .unwrap_or(0);
    let target_rank = candidates
        .iter()
        .position(|candidate| candidate.token == target)
        .map(|index| index + 1)
        .unwrap_or_else(|| candidates.len().saturating_add(1));
    let target_logit = row.logits_q8[usize::from(target)];
    let best_competing_logit = candidates
        .iter()
        .find(|candidate| candidate.token != target)
        .map(|candidate| candidate.logit_q8)
        .unwrap_or(target_logit);
    EvalTokenDiagnostics {
        predicted,
        target_rank,
        target_margin_q8: target_logit.saturating_sub(best_competing_logit),
    }
}

fn eval_allowed_tokens(phase: EvalPhase, text_token_profile: TextTokenProfile) -> Vec<u8> {
    match phase {
        EvalPhase::Special => vec![PROMPT, TEXT, IMAGE, EOS],
        EvalPhase::Prompt | EvalPhase::Text => allowed_text_tokens(false, text_token_profile),
        EvalPhase::Image => (0..IMAGE_BINS)
            .map(|bin| IMAGE_BASE.saturating_add(bin))
            .collect(),
    }
}

fn conditioning_score(prompt: &str, example_prompt: &str, primary_name: &str) -> usize {
    let prompt_key = normalize_key(prompt);
    let example_key = normalize_key(example_prompt);
    let name_key = normalize_key(primary_name);
    let mut score = 0_usize;
    if prompt_key == example_key {
        score = score.saturating_add(1_000_000);
    }
    if !name_key.is_empty() && prompt_key.split_whitespace().any(|word| word == name_key) {
        score = score.saturating_add(100_000);
    }
    for word in prompt_key.split_whitespace() {
        if word.len() >= 3
            && example_key
                .split_whitespace()
                .any(|candidate| candidate == word)
        {
            score = score.saturating_add(100);
        }
    }
    score
}

fn sample_model(
    model: &SolomonAttentionModel,
    prompt: &str,
    options: SampleOptions,
    conditioning: Option<&ConditioningMatch>,
    text_prior: Option<&TextPrior>,
    image_prior_tokens: Option<&[u8]>,
    text_prefix: &str,
) -> Result<Sample, Box<dyn std::error::Error>> {
    let normalized_prompt = normalize_text(prompt);
    let seed = options.sample_seed ^ hash_text(&normalized_prompt) ^ model.model_hash()?;
    let decode_options = DecodeOptions {
        seed,
        repeat_run_cap: options.repeat_run_cap,
        no_repeat_ngram: options.no_repeat_ngram,
        expected_token_boost_q8: options.conditioning_boost_q8,
        text_prior_boost_q8: options.text_prior_boost_q8,
        text_prior_strict: options.text_prior_strict,
        text_chunk_boost_q8: options.text_chunk_boost_q8,
        decode_logit_delta: options.decode_logit_delta,
        prompt_name_opening_tokens: if options.prompt_name_opening_prior {
            prompt_name_opening_tokens(&normalized_prompt, model.text_token_profile)
        } else {
            Vec::new()
        },
        suppress_name_chunks_after_opening: options.suppress_name_chunks_after_opening,
        top_k: options.top_k,
    };
    let baseline_transformer = options.decode_logit_delta.then(|| {
        MiniTransformerMlpModel::new_initial_with_seq_len(model.transformer.context_seq_len)
    });
    let mut generated_tokens = Vec::new();
    generated_tokens.push(BOS);
    generated_tokens.push(PROMPT);
    generated_tokens.extend(encode_text_tokens(
        &normalized_prompt,
        model.text_token_profile,
    ));
    generated_tokens.push(TEXT);
    let text_prior_start_tokens = text_prior.map(|prior| prior.start_tokens_for_prompt(prompt));

    let mut text_tokens = Vec::new();
    let prefix_tokens = encode_text_prefix_tokens(text_prefix, model.text_token_profile);
    if prefix_tokens.len() > options.max_text_tokens {
        return Err(format!(
            "--text-prefix encodes to {} tokens, above --max-text-tokens {}",
            prefix_tokens.len(),
            options.max_text_tokens
        )
        .into());
    }
    generated_tokens.extend(prefix_tokens.iter().copied());
    text_tokens.extend(prefix_tokens.iter().copied());
    for step in prefix_tokens.len()..options.max_text_tokens {
        if conditioning.is_some_and(|value| step >= value.text_tokens.len()) {
            break;
        }
        let allow_stop = step >= options.min_text_tokens;
        let expected_token = conditioning.and_then(|value| value.text_tokens.get(step).copied());
        let token = next_token(
            model,
            NextTokenRequest {
                history: &generated_tokens,
                phase: Phase::Text,
                allow_stop,
                step,
                text_token_profile: model.text_token_profile,
                expected_token,
                text_prior,
                text_prior_start_tokens,
            },
            &decode_options,
            baseline_transformer.as_ref(),
        )?;
        if allow_stop && (token == IMAGE || token == EOS) {
            break;
        }
        if is_text_token(token) {
            generated_tokens.push(token);
            text_tokens.push(token);
        }
    }
    generated_tokens.push(IMAGE);

    let mut image_bins = [0_u8; SIGNATURE_BINS];
    for (index, bin) in image_bins.iter_mut().enumerate() {
        let expected_token = conditioning
            .and_then(|value| value.image_tokens.get(index).copied())
            .or_else(|| image_prior_tokens.and_then(|tokens| tokens.get(index).copied()));
        let image_decode_options = DecodeOptions {
            repeat_run_cap: 0,
            ..decode_options.clone()
        };
        let token = next_token(
            model,
            NextTokenRequest {
                history: &generated_tokens,
                phase: Phase::Image,
                allow_stop: false,
                step: options.max_text_tokens.saturating_add(index),
                text_token_profile: model.text_token_profile,
                expected_token,
                text_prior: None,
                text_prior_start_tokens: None,
            },
            &image_decode_options,
            baseline_transformer.as_ref(),
        )?;
        let image_token = if is_image_token(token) {
            token
        } else {
            IMAGE_BASE
        };
        *bin = image_token.saturating_sub(IMAGE_BASE);
        generated_tokens.push(image_token);
    }
    generated_tokens.push(EOS);

    Ok(Sample {
        prompt: normalized_prompt,
        text: decode_text_tokens(&text_tokens, model.text_token_profile),
        image_bins,
        generated_tokens,
    })
}

fn next_token(
    model: &SolomonAttentionModel,
    request: NextTokenRequest<'_>,
    options: &DecodeOptions,
    baseline_transformer: Option<&MiniTransformerMlpModel>,
) -> Result<u8, Box<dyn std::error::Error>> {
    let row = next_token_row_with_context_backoff(model, request.history)?;
    let baseline_row = if options.decode_logit_delta {
        Some(next_token_row_with_context_backoff_for_transformer(
            baseline_transformer
                .ok_or("--decode-logit-delta requires an initial baseline model")?,
            model.attention_kind,
            model.position_policy,
            request.history,
        )?)
    } else {
        None
    };
    let mut candidates = Vec::new();
    for token in allowed_tokens(
        request.phase,
        request.allow_stop,
        request.text_token_profile,
    ) {
        let index = usize::from(token);
        let logit_q8 = baseline_row
            .as_ref()
            .map(|baseline| row.logits_q8[index].saturating_sub(baseline.logits_q8[index]))
            .unwrap_or(row.logits_q8[index]);
        candidates.push(Candidate {
            token,
            logit_q8,
            probability_q15: i32::from(row.probabilities_q15[index]).max(0),
        });
    }
    if let Some(expected_token) = request.expected_token {
        apply_expected_token_boost(
            &mut candidates,
            expected_token,
            options.expected_token_boost_q8,
        );
    }
    if request.phase == Phase::Text
        && request.text_token_profile == TextTokenProfile::Chunked
        && options.text_chunk_boost_q8 != 0
    {
        apply_text_chunk_boost(&mut candidates, options.text_chunk_boost_q8);
    }
    if request.phase == Phase::Text
        && request.text_token_profile == TextTokenProfile::Chunked
        && options.suppress_name_chunks_after_opening
        && generated_text_is_after_opening(request.history)
    {
        suppress_name_chunk_candidates(&mut candidates);
    }
    if request.phase == Phase::Text && request.expected_token.is_none() {
        if !options.prompt_name_opening_tokens.is_empty() {
            apply_prompt_name_opening_prior(
                &mut candidates,
                request.history,
                &options.prompt_name_opening_tokens,
            );
        }
        let text_prior_matched = if let Some(text_prior) = request.text_prior {
            apply_text_prior(
                &mut candidates,
                request.history,
                text_prior,
                request
                    .text_prior_start_tokens
                    .unwrap_or(&text_prior.start_tokens),
                options.text_prior_boost_q8,
                options.text_prior_strict,
            )
        } else {
            false
        };
        if !(options.text_prior_strict && text_prior_matched) {
            let text_context = generated_text_context(request.history);
            apply_repeat_run_cap(&mut candidates, &text_context, options.repeat_run_cap);
            apply_no_repeat_ngram(&mut candidates, &text_context, options.no_repeat_ngram);
        }
    }
    Ok(choose_candidate(&mut candidates, options.seed, request.step, options.top_k).token)
}

fn next_token_row_with_context_backoff(
    model: &SolomonAttentionModel,
    history: &[u8],
) -> Result<MiniTransformerNextTokenRow, Box<dyn std::error::Error>> {
    next_token_row_with_context_backoff_for_transformer(
        &model.transformer,
        model.attention_kind,
        model.position_policy,
        history,
    )
}

fn next_token_row_with_context_backoff_for_transformer(
    transformer: &MiniTransformerMlpModel,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
    history: &[u8],
) -> Result<MiniTransformerNextTokenRow, Box<dyn std::error::Error>> {
    let max_context_len = transformer.context_seq_len.min(history.len());
    let mut padded_context = PaddedContext::new(transformer.context_seq_len);
    let training_context = padded_context.window(history, transformer.context_seq_len);
    if let Ok(row) = mini_transformer_next_token_row_with_attention_kind_position_policy(
        transformer,
        training_context,
        attention_kind,
        position_policy,
    ) {
        return Ok(row);
    }
    for context_len in (1..=max_context_len).rev() {
        let context = &history[history.len() - context_len..];
        if let Ok(row) = mini_transformer_next_token_row_with_attention_kind_position_policy(
            transformer,
            context,
            attention_kind,
            position_policy,
        ) {
            return Ok(row);
        }
    }
    Err("no valid attention context for generated history".into())
}

fn apply_text_chunk_boost(candidates: &mut [Candidate], boost_q8: i32) {
    for candidate in candidates {
        if is_text_chunk_token(candidate.token) {
            candidate.logit_q8 = candidate.logit_q8.saturating_add(boost_q8);
            candidate.probability_q15 = candidate.probability_q15.saturating_add(boost_q8.max(0));
        }
    }
}

fn suppress_name_chunk_candidates(candidates: &mut Vec<Candidate>) {
    let original_len = candidates.len();
    candidates.retain(|candidate| !is_name_text_chunk_token(candidate.token));
    if candidates.is_empty() && original_len > 0 {
        candidates.push(Candidate {
            token: TEXT_BASE + b' ',
            logit_q8: i32::MIN,
            probability_q15: 0,
        });
    }
}

fn apply_expected_token_boost(
    candidates: &mut [Candidate],
    expected_token: u8,
    expected_token_boost_q8: i32,
) {
    if expected_token_boost_q8 <= 0 {
        return;
    }
    for candidate in candidates {
        if candidate.token == expected_token {
            candidate.logit_q8 = candidate.logit_q8.saturating_add(expected_token_boost_q8);
            candidate.probability_q15 = candidate
                .probability_q15
                .saturating_add(expected_token_boost_q8.max(0));
            return;
        }
    }
}

fn apply_repeat_run_cap(candidates: &mut Vec<Candidate>, history: &[u8], repeat_run_cap: usize) {
    if repeat_run_cap == 0 {
        return;
    }
    let Some(&last_token) = history.last() else {
        return;
    };
    if !is_text_token(last_token) {
        return;
    }
    let run_len = history
        .iter()
        .rev()
        .take_while(|&&token| token == last_token)
        .count();
    if run_len < repeat_run_cap {
        return;
    }
    let original_len = candidates.len();
    candidates.retain(|candidate| candidate.token != last_token);
    if candidates.is_empty() && original_len > 0 {
        candidates.push(Candidate {
            token: last_token,
            logit_q8: i32::MIN,
            probability_q15: 0,
        });
    }
}

fn apply_no_repeat_ngram(candidates: &mut Vec<Candidate>, history: &[u8], no_repeat_ngram: usize) {
    if no_repeat_ngram < 2 || history.len() + 1 < no_repeat_ngram {
        return;
    }
    let prefix_len = no_repeat_ngram - 1;
    let prefix_start = history.len() - prefix_len;
    let prefix = &history[prefix_start..];
    let original_len = candidates.len();
    candidates.retain(|candidate| {
        let mut candidate_ngram = Vec::with_capacity(no_repeat_ngram);
        candidate_ngram.extend_from_slice(prefix);
        candidate_ngram.push(candidate.token);
        !history
            .windows(no_repeat_ngram)
            .any(|window| window == candidate_ngram.as_slice())
    });
    if candidates.is_empty() && original_len > 0 {
        candidates.push(Candidate {
            token: history.last().copied().unwrap_or(TEXT_BASE + b' '),
            logit_q8: i32::MIN,
            probability_q15: 0,
        });
    }
}

fn apply_prompt_name_opening_prior(
    candidates: &mut Vec<Candidate>,
    history: &[u8],
    opening_tokens: &[u8],
) {
    if opening_tokens.is_empty() {
        return;
    }
    let text_context = generated_text_context(history);
    if text_context.len() >= opening_tokens.len()
        || !opening_tokens[..text_context.len()].eq(text_context.as_slice())
    {
        return;
    };
    let expected = opening_tokens[text_context.len()];
    if candidates
        .iter()
        .any(|candidate| candidate.token == expected)
    {
        candidates.retain(|candidate| candidate.token == expected);
    }
}

fn apply_text_prior(
    candidates: &mut Vec<Candidate>,
    history: &[u8],
    prior: &TextPrior,
    start_tokens: &[u8],
    boost_q8: i32,
    strict: bool,
) -> bool {
    if candidates.is_empty() || (!strict && boost_q8 <= 0) {
        return false;
    }
    let text_context = generated_text_context(history);
    if text_context.len() < start_tokens.len() {
        let expected = start_tokens[text_context.len()];
        let mut matched = false;
        for candidate in candidates.iter_mut() {
            if candidate.token == expected {
                matched = true;
                candidate.logit_q8 = candidate.logit_q8.saturating_add(boost_q8.max(1));
                candidate.probability_q15 = candidate
                    .probability_q15
                    .saturating_add(boost_q8.max(1).max(0));
            }
        }
        if strict {
            candidates.retain(|candidate| candidate.token == expected);
        }
        if matched && !candidates.is_empty() {
            return true;
        }
    }
    if text_context.is_empty()
        && let Some(counts) = prior.transitions.get(&vec![TEXT])
        && apply_text_prior_counts(candidates, counts, boost_q8, strict)
    {
        return true;
    }
    let max_order = prior.order.min(text_context.len());
    let min_order = prior.min_order.min(max_order);
    for order in (min_order..=max_order).rev() {
        let key = text_context[text_context.len() - order..].to_vec();
        let Some(counts) = prior.transitions.get(&key) else {
            continue;
        };
        if apply_text_prior_counts(candidates, counts, boost_q8, strict) {
            return true;
        }
    }
    false
}

fn apply_text_prior_counts(
    candidates: &mut Vec<Candidate>,
    counts: &[u32; VOCAB_LEN],
    boost_q8: i32,
    strict: bool,
) -> bool {
    let mut matched = 0_usize;
    for candidate in candidates.iter_mut() {
        let count = counts[usize::from(candidate.token)];
        if count == 0 {
            continue;
        }
        matched += 1;
        let count_boost = i32::try_from(count.min(4096)).unwrap_or(4096) * 256;
        candidate.logit_q8 = candidate
            .logit_q8
            .saturating_add(boost_q8)
            .saturating_add(count_boost);
        candidate.probability_q15 = candidate
            .probability_q15
            .saturating_add(boost_q8.max(0))
            .saturating_add(count_boost.max(0));
    }
    if matched > 0 && strict {
        candidates.retain(|candidate| counts[usize::from(candidate.token)] > 0);
    }
    matched > 0
}

fn generated_text_context(history: &[u8]) -> Vec<u8> {
    let start = history
        .iter()
        .rposition(|&token| token == TEXT)
        .map(|index| index + 1)
        .unwrap_or(history.len());
    history[start..]
        .iter()
        .copied()
        .take_while(|&token| token != IMAGE && token != EOS)
        .filter(|&token| is_text_token(token))
        .collect()
}

fn generated_text_is_after_opening(history: &[u8]) -> bool {
    generated_text_context(history)
        .windows(2)
        .any(|window| window == [TEXT_CHUNK_BASE + 1, TEXT_CHUNK_BASE + 2])
}

fn allowed_tokens(phase: Phase, allow_stop: bool, text_token_profile: TextTokenProfile) -> Vec<u8> {
    match phase {
        Phase::Text => {
            let mut tokens = allowed_text_tokens(false, text_token_profile);
            if allow_stop {
                tokens.push(IMAGE);
                tokens.push(EOS);
            }
            tokens
        }
        Phase::Image => (0..IMAGE_BINS)
            .map(|bin| IMAGE_BASE.saturating_add(bin))
            .collect(),
    }
}

fn allowed_text_tokens(include_stop: bool, text_token_profile: TextTokenProfile) -> Vec<u8> {
    let mut tokens: Vec<u8> = (32_u8..=126)
        .map(|byte| TEXT_BASE.saturating_add(byte))
        .collect();
    if text_token_profile == TextTokenProfile::Chunked {
        for chunk_index in 0..TEXT_CHUNKS.len() {
            tokens.push(TEXT_CHUNK_BASE.saturating_add(u8::try_from(chunk_index).unwrap_or(0)));
        }
    }
    if include_stop {
        tokens.push(IMAGE);
        tokens.push(EOS);
    }
    tokens
}

fn choose_candidate(
    candidates: &mut [Candidate],
    seed: u64,
    step: usize,
    top_k: usize,
) -> Candidate {
    candidates.sort_by(|left, right| {
        right
            .logit_q8
            .cmp(&left.logit_q8)
            .then_with(|| right.probability_q15.cmp(&left.probability_q15))
            .then_with(|| left.token.cmp(&right.token))
    });
    let limit = top_k.max(1).min(candidates.len());
    if limit <= 1 {
        return candidates[0];
    }
    let total = candidates.iter().take(limit).fold(0_u64, |acc, candidate| {
        acc.saturating_add(u64::try_from(candidate.probability_q15).unwrap_or(0))
    });
    if total == 0 {
        let index = usize::try_from(mix64(seed ^ step as u64) % u64::try_from(limit).unwrap_or(1))
            .unwrap_or(0);
        return candidates[index];
    }
    let mut draw = mix64(seed ^ step as u64) % total;
    for candidate in candidates.iter().take(limit) {
        let weight = u64::try_from(candidate.probability_q15).unwrap_or(0);
        if draw < weight {
            return *candidate;
        }
        draw = draw.saturating_sub(weight);
    }
    candidates[0]
}

impl SolomonAttentionModel {
    fn try_to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut out = self.bytes_without_hash()?;
        out.extend_from_slice(&self.model_hash()?.to_le_bytes());
        Ok(out)
    }

    fn bytes_without_hash(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let transformer_bytes = self.transformer.try_to_bytes()?;
        let mut out = Vec::with_capacity(96 + transformer_bytes.len());
        out.extend_from_slice(MODEL_MAGIC);
        push_u32(&mut out, MODEL_VERSION);
        push_u32(&mut out, u32::from(PAD));
        push_u32(&mut out, u32::try_from(VOCAB_SIZE)?);
        push_u32(&mut out, u32::from(TEXT_BASE));
        push_u32(&mut out, u32::from(TEXT_COUNT));
        push_u32(&mut out, u32::from(IMAGE_BASE));
        push_u32(&mut out, u32::from(IMAGE_BINS));
        push_u32(&mut out, u32::try_from(SIGNATURE_GRID)?);
        push_u32(&mut out, attention_kind_code(self.attention_kind));
        push_u32(&mut out, position_policy_code(self.position_policy));
        push_u32(&mut out, text_token_profile_code(self.text_token_profile));
        push_text_memory(&mut out, self.text_memory.as_ref())?;
        push_u32(&mut out, u32::try_from(self.transformer.context_seq_len)?);
        push_u64(&mut out, self.token_count);
        push_u64(&mut out, self.token_hash);
        push_u64(&mut out, self.transformer.model_hash());
        push_u64(&mut out, u64::try_from(transformer_bytes.len())?);
        out.extend_from_slice(&transformer_bytes);
        Ok(out)
    }

    fn model_hash(&self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(hash_bytes(&self.bytes_without_hash()?))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < MODEL_MAGIC.len() + 8 || &bytes[..MODEL_MAGIC.len()] != MODEL_MAGIC {
            return Err("not an NSRLLMM1 model".into());
        }
        let expected_hash = read_u64_at_end(bytes)?;
        let actual_hash = hash_bytes(&bytes[..bytes.len() - 8]);
        if expected_hash != actual_hash {
            return Err(format!(
                "NSRLLMM1 hash mismatch: expected 0x{expected_hash:016x}, got 0x{actual_hash:016x}"
            )
            .into());
        }
        let mut cursor = Cursor::new(&bytes[MODEL_MAGIC.len()..bytes.len() - 8]);
        let version = cursor.read_u32()?;
        if version == 0 || version > MODEL_VERSION {
            return Err(format!("unsupported NSRLLMM1 version {version}").into());
        }
        expect_u32(&mut cursor, u32::from(PAD), "pad token")?;
        if version == 1 {
            expect_u32(&mut cursor, LEGACY_VOCAB_SIZE, "vocab size")?;
        } else {
            expect_u32(&mut cursor, u32::try_from(VOCAB_SIZE)?, "vocab size")?;
        }
        expect_u32(&mut cursor, u32::from(TEXT_BASE), "text base")?;
        expect_u32(&mut cursor, u32::from(TEXT_COUNT), "text count")?;
        expect_u32(&mut cursor, u32::from(IMAGE_BASE), "image base")?;
        expect_u32(&mut cursor, u32::from(IMAGE_BINS), "image bins")?;
        expect_u32(
            &mut cursor,
            u32::try_from(SIGNATURE_GRID)?,
            "signature grid",
        )?;
        let attention_kind = attention_kind_from_code(cursor.read_u32()?)?;
        let position_policy = position_policy_from_code(cursor.read_u32()?)?;
        let text_token_profile = if version == 1 {
            TextTokenProfile::Char
        } else {
            text_token_profile_from_code(cursor.read_u32()?)?
        };
        let text_memory = if version >= 3 {
            cursor.read_text_memory(version)?
        } else {
            None
        };
        let context_seq_len = usize::try_from(cursor.read_u32()?)?;
        let token_count = cursor.read_u64()?;
        let token_hash = cursor.read_u64()?;
        let expected_inner_hash = cursor.read_u64()?;
        let transformer_len = usize::try_from(cursor.read_u64()?)?;
        let transformer_bytes = cursor.read_exact(transformer_len)?;
        if !cursor.is_empty() {
            return Err("trailing bytes in NSRLLMM1 body".into());
        }
        let transformer = MiniTransformerMlpModel::from_bytes(transformer_bytes)?;
        if transformer.context_seq_len != context_seq_len {
            return Err("NSRLLMM1 context length mismatch".into());
        }
        if transformer.model_hash() != expected_inner_hash {
            return Err("NSRLLMM1 inner model hash mismatch".into());
        }
        Ok(Self {
            token_count,
            token_hash,
            attention_kind,
            position_policy,
            text_token_profile,
            text_memory,
            transformer,
        })
    }
}

fn attention_kind_code(kind: MiniTransformerAttentionKind) -> u32 {
    match kind {
        MiniTransformerAttentionKind::Base2Softmax => 0,
        MiniTransformerAttentionKind::Linear => 1,
        MiniTransformerAttentionKind::LinearStreamingNope => 2,
        MiniTransformerAttentionKind::LinearStreamingTttNope => 3,
    }
}

fn attention_kind_from_code(
    code: u32,
) -> Result<MiniTransformerAttentionKind, Box<dyn std::error::Error>> {
    match code {
        0 => Ok(MiniTransformerAttentionKind::Base2Softmax),
        1 => Ok(MiniTransformerAttentionKind::Linear),
        2 => Ok(MiniTransformerAttentionKind::LinearStreamingNope),
        3 => Ok(MiniTransformerAttentionKind::LinearStreamingTttNope),
        _ => Err(format!("unsupported attention kind code {code}").into()),
    }
}

fn position_policy_code(policy: MiniTransformerPositionPolicy) -> u32 {
    match policy {
        MiniTransformerPositionPolicy::LearnedAbsolute => 0,
        MiniTransformerPositionPolicy::Nope => 1,
    }
}

fn position_policy_from_code(
    code: u32,
) -> Result<MiniTransformerPositionPolicy, Box<dyn std::error::Error>> {
    match code {
        0 => Ok(MiniTransformerPositionPolicy::LearnedAbsolute),
        1 => Ok(MiniTransformerPositionPolicy::Nope),
        _ => Err(format!("unsupported position policy code {code}").into()),
    }
}

fn text_token_profile_code(profile: TextTokenProfile) -> u32 {
    match profile {
        TextTokenProfile::Char => 0,
        TextTokenProfile::Chunked => 1,
    }
}

fn text_token_profile_from_code(code: u32) -> Result<TextTokenProfile, Box<dyn std::error::Error>> {
    match code {
        0 => Ok(TextTokenProfile::Char),
        1 => Ok(TextTokenProfile::Chunked),
        _ => Err(format!("unsupported text token profile code {code}").into()),
    }
}

fn is_printable_text_token(token: u8) -> bool {
    (TEXT_BASE + 32..=TEXT_BASE + 126).contains(&token)
}

fn is_text_chunk_token(token: u8) -> bool {
    let start = usize::from(TEXT_CHUNK_BASE);
    let token = usize::from(token);
    token >= start && token < start.saturating_add(TEXT_CHUNKS.len())
}

fn is_name_text_chunk_token(token: u8) -> bool {
    is_text_chunk_token(token)
        && usize::from(token.saturating_sub(TEXT_CHUNK_BASE)) >= TEXT_CHUNK_NAME_START
}

fn is_text_token(token: u8) -> bool {
    is_printable_text_token(token) || is_text_chunk_token(token)
}

fn is_image_token(token: u8) -> bool {
    (IMAGE_BASE..IMAGE_BASE + IMAGE_BINS).contains(&token)
}

fn encode_text_tokens(text: &str, profile: TextTokenProfile) -> Vec<u8> {
    let normalized = normalize_text(text);
    encode_normalized_text_tokens(&normalized, profile)
}

fn encode_text_prefix_tokens(text: &str, profile: TextTokenProfile) -> Vec<u8> {
    let mut normalized = normalize_text(text);
    if text.ends_with(' ') && !normalized.ends_with(' ') && !normalized.is_empty() {
        normalized.push(' ');
    }
    encode_normalized_text_tokens(&normalized, profile)
}

fn prompt_name_opening_tokens(prompt: &str, profile: TextTokenProfile) -> Vec<u8> {
    let Some(name) = prompt_spirit_name(prompt) else {
        return Vec::new();
    };
    encode_text_prefix_tokens(&format!("Solomon selects {name}: He "), profile)
}

fn prompt_spirit_name(prompt: &str) -> Option<&'static str> {
    let prompt_key = normalize_key(prompt);
    TEXT_CHUNKS
        .get(TEXT_CHUNK_NAME_START..)
        .unwrap_or(&[])
        .iter()
        .copied()
        .find(|name| {
            let name_key = normalize_key(name);
            !name_key.is_empty() && prompt_contains_phrase(&prompt_key, &name_key)
        })
}

fn encode_normalized_text_tokens(normalized: &str, profile: TextTokenProfile) -> Vec<u8> {
    let bytes = normalized.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut index = 0_usize;
    while index < bytes.len() {
        if profile == TextTokenProfile::Chunked
            && let Some((chunk_index, chunk)) = match_text_chunk(bytes, index)
        {
            tokens.push(TEXT_CHUNK_BASE.saturating_add(u8::try_from(chunk_index).unwrap_or(0)));
            index += chunk.len();
            continue;
        }
        tokens.push(TEXT_BASE.saturating_add(bytes[index].min(127)));
        index += 1;
    }
    tokens
}

fn decode_text_tokens(tokens: &[u8], profile: TextTokenProfile) -> String {
    let mut out = String::new();
    for &token in tokens {
        if is_printable_text_token(token) {
            out.push(char::from(token - TEXT_BASE));
        } else if profile == TextTokenProfile::Chunked && is_text_chunk_token(token) {
            let chunk_index = usize::from(token.saturating_sub(TEXT_CHUNK_BASE));
            if let Some(chunk) = TEXT_CHUNKS.get(chunk_index) {
                out.push_str(chunk);
            }
        }
    }
    compact_spaces(&out)
}

fn sanitize_source_text_tokens(tokens: &[u8], profile: TextTokenProfile) -> Vec<u8> {
    encode_text_tokens(
        &normalize_text(&decode_text_tokens(tokens, profile)),
        profile,
    )
}

fn match_text_chunk(bytes: &[u8], index: usize) -> Option<(usize, &'static [u8])> {
    let mut best: Option<(usize, &'static [u8])> = None;
    for (chunk_index, chunk) in TEXT_CHUNKS.iter().enumerate() {
        let chunk_bytes = chunk.as_bytes();
        if !bytes[index..].starts_with(chunk_bytes) {
            continue;
        }
        if best.is_none_or(|(_, best_chunk)| chunk_bytes.len() > best_chunk.len()) {
            best = Some((chunk_index, chunk_bytes));
        }
    }
    best
}

fn normalize_text(text: &str) -> String {
    let ascii = text.chars().map(normalize_char).collect::<String>();
    compact_spaces(&strip_numeric_footnote_refs(&ascii))
}

fn normalize_key(text: &str) -> String {
    normalize_text(text).to_ascii_lowercase()
}

fn strip_numeric_footnote_refs(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0_usize;
    while index < bytes.len() {
        if bytes[index] == b'[' {
            let digit_start = index + 1;
            let mut end = digit_start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > digit_start && end < bytes.len() && bytes[end] == b']' {
                out.push(' ');
                index = end + 1;
                continue;
            }
        }
        out.push(char::from(bytes[index]));
        index += 1;
    }
    out
}

fn normalize_char(ch: char) -> char {
    match ch {
        ch if ch.is_ascii_graphic() || ch == ' ' => ch,
        'Á' | 'À' | 'Â' | 'Ä' | 'Ã' | 'Å' | 'Ā' | 'Ă' | 'Ą' => 'A',
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'Ç' | 'Ć' | 'Ĉ' | 'Ċ' | 'Č' => 'C',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'Ð' | 'Ď' | 'Đ' => 'D',
        'ð' | 'ď' | 'đ' => 'd',
        'É' | 'È' | 'Ê' | 'Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => 'E',
        'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'Í' | 'Ì' | 'Î' | 'Ï' | 'Ĩ' | 'Ī' | 'Ĭ' | 'Į' | 'İ' => 'I',
        'í' | 'ì' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'Ñ' | 'Ń' | 'Ņ' | 'Ň' => 'N',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' | 'Ō' | 'Ŏ' | 'Ő' => 'O',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ō' | 'ŏ' | 'ő' => 'o',
        'Ŕ' | 'Ŗ' | 'Ř' => 'R',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'Ś' | 'Ŝ' | 'Ş' | 'Š' => 'S',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'Ú' | 'Ù' | 'Û' | 'Ü' | 'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => 'U',
        'ú' | 'ù' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'Ý' | 'Ŷ' | 'Ÿ' => 'Y',
        'ý' | 'ŷ' | 'ÿ' => 'y',
        'Ž' | 'Ź' | 'Ż' => 'Z',
        'ž' | 'ź' | 'ż' => 'z',
        'Æ' => 'A',
        'æ' => 'a',
        'Œ' => 'O',
        'œ' => 'o',
        'Þ' => 'T',
        'þ' => 't',
        'ß' => 's',
        '‘' | '’' => '\'',
        '“' | '”' => '"',
        '–' | '—' => '-',
        _ => ' ',
    }
}

fn compact_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn json_string_field(line: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":");
    let start = line.find(&key)? + key.len();
    let bytes = line.as_bytes();
    let mut index = start;
    while bytes
        .get(index)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        index += 1;
    }
    if bytes.get(index).copied()? != b'"' {
        return None;
    }
    index += 1;
    let mut out = String::new();
    while let Some(&byte) = bytes.get(index) {
        index += 1;
        match byte {
            b'"' => return Some(out),
            b'\\' => {
                let escaped = *bytes.get(index)?;
                index += 1;
                match escaped {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    _ => return None,
                }
            }
            value => out.push(char::from(value)),
        }
    }
    None
}

fn json_usize_field(line: &str, field: &str) -> Option<usize> {
    let key = format!("\"{field}\":");
    let start = line.find(&key)? + key.len();
    let bytes = line.as_bytes();
    let mut index = start;
    while bytes
        .get(index)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        index += 1;
    }
    let digit_start = index;
    while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
        index += 1;
    }
    if index == digit_start {
        return None;
    }
    line[digit_start..index].parse().ok()
}

fn image_ink_bytes(image_bins: &[u8; SIGNATURE_BINS]) -> Vec<u8> {
    image_bins
        .iter()
        .map(|&bin| bin.saturating_mul(17))
        .collect()
}

fn image_pgm_bytes(image_bins: &[u8; SIGNATURE_BINS]) -> Vec<u8> {
    let image_size = 128_usize;
    let scale = image_size / SIGNATURE_GRID;
    let mut out = format!("P5\n{} {}\n255\n", image_size, image_size).into_bytes();
    for y in 0..image_size {
        let gy = y / scale;
        for x in 0..image_size {
            let gx = x / scale;
            let ink = image_bins[gy * SIGNATURE_GRID + gx].saturating_mul(17);
            out.push(u8::MAX.saturating_sub(ink));
        }
    }
    out
}

fn image_token_tsv(image_bins: &[u8; SIGNATURE_BINS]) -> String {
    let mut out = String::new();
    for y in 0..SIGNATURE_GRID {
        for x in 0..SIGNATURE_GRID {
            if x > 0 {
                out.push('\t');
            }
            out.push_str(&image_bins[y * SIGNATURE_GRID + x].to_string());
        }
        out.push('\n');
    }
    out
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_text_memory(
    out: &mut Vec<u8>,
    memory: Option<&TextMemory>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(memory) = memory else {
        push_u32(out, 0);
        return Ok(());
    };
    push_u32(out, 1);
    push_u32(out, u32::try_from(memory.order)?);
    push_u32(out, u32::try_from(memory.examples.len())?);
    for example in &memory.examples {
        push_string(out, &example.primary_name)?;
        push_string(out, &example.prompt)?;
        push_u8_vec(out, &example.text_tokens)?;
        push_u8_vec(out, &example.image_tokens)?;
    }
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), Box<dyn std::error::Error>> {
    push_u8_vec(out, value.as_bytes())
}

fn push_u8_vec(out: &mut Vec<u8>, value: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    push_u32(out, u32::try_from(value.len())?);
    out.extend_from_slice(value);
    Ok(())
}

fn expect_u32(
    cursor: &mut Cursor<'_>,
    expected: u32,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = cursor.read_u32()?;
    if actual != expected {
        return Err(format!("NSRLLMM1 {label} mismatch: expected {expected}, got {actual}").into());
    }
    Ok(())
}

fn read_u64_at_end(bytes: &[u8]) -> Result<u64, Box<dyn std::error::Error>> {
    if bytes.len() < 8 {
        return Err("buffer too short for u64".into());
    }
    let offset = bytes.len() - 8;
    Ok(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_text_memory(
        &mut self,
        version: u32,
    ) -> Result<Option<TextMemory>, Box<dyn std::error::Error>> {
        let present = self.read_u32()?;
        if present == 0 {
            return Ok(None);
        }
        if present != 1 {
            return Err(format!("invalid NSRLLMM1 text memory marker {present}").into());
        }
        let order = usize::try_from(self.read_u32()?)?;
        let example_count = usize::try_from(self.read_u32()?)?;
        let mut examples = Vec::with_capacity(example_count);
        for _ in 0..example_count {
            examples.push(TextMemoryExample {
                primary_name: self.read_string()?,
                prompt: self.read_string()?,
                text_tokens: self.read_u8_vec()?,
                image_tokens: if version >= 4 {
                    self.read_u8_vec()?
                } else {
                    Vec::new()
                },
            });
        }
        Ok(Some(TextMemory { order, examples }))
    }

    fn read_string(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let bytes = self.read_u8_vec()?;
        Ok(String::from_utf8(bytes)?)
    }

    fn read_u8_vec(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let count = usize::try_from(self.read_u32()?)?;
        Ok(self.read_exact(count)?.to_vec())
    }

    fn read_exact(&mut self, count: usize) -> Result<&'a [u8], Box<dyn std::error::Error>> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or("cursor offset overflow")?;
        if end > self.bytes.len() {
            return Err("truncated NSRLLMM1 model".into());
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
}

fn hash_text(text: &str) -> u64 {
    hash_bytes(text.as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value.is_control() => {
                use std::fmt::Write;
                let _ = write!(&mut out, "\\u{:04x}", value as u32);
            }
            value => out.push(value),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_round_trips_and_samples_joint_text_and_image() {
        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: Some(TextMemory {
                order: 2,
                examples: vec![TextMemoryExample {
                    primary_name: String::from("Bael"),
                    prompt: String::from("seal of Bael"),
                    text_tokens: vec![TEXT_CHUNK_BASE, TEXT_BASE + b'B'],
                    image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
                }],
            }),
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let bytes = model.try_to_bytes().unwrap();
        let loaded = SolomonAttentionModel::from_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);

        let sample = sample_model(
            &loaded,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: 8,
                min_text_tokens: 4,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            None,
            None,
            "",
        )
        .unwrap();
        assert_eq!(sample.image_bins.len(), SIGNATURE_BINS);
        assert!(sample.image_bins.iter().all(|&bin| bin < IMAGE_BINS));
        assert!(sample.generated_tokens.contains(&IMAGE));
        assert_eq!(sample.generated_tokens.last().copied(), Some(EOS));
    }

    #[test]
    fn parses_train_window_offset() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--seq-len",
                "32",
                "--stride",
                "8",
                "--window-offset",
                "7",
                "--learning-rate",
                "2",
                "--output-lr-shift",
                "19",
                "--mlp-lr-shift",
                "17",
                "--embed-lr-shift",
                "15",
                "--attention-lr-shift",
                "25",
                "--attention-q-lr-shift",
                "20",
                "--attention-qk-lr-shift",
                "21",
                "--target-frequency-cap",
                "8",
                "--target-frequency-min-weight-q15",
                "2048",
                "--argmax-margin-weight-q15",
                "1024",
                "--target-segment",
                "generated-text",
                "--target-phase",
                "image",
                "--text-token-profile",
                "chunked",
                "--zero-output-head-init",
                "--solomon-name-copy-init",
                "--reject-loss-regression",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.command, Command::Train);
        assert_eq!(config.seq_len, 32);
        assert_eq!(config.stride, 8);
        assert_eq!(config.window_offset, 7);
        assert_eq!(config.learning_rate, 2);
        assert_eq!(config.output_learning_rate_shift, 19);
        assert_eq!(config.mlp_learning_rate_shift, 17);
        assert_eq!(config.embedding_learning_rate_shift, 15);
        assert_eq!(config.attention_learning_rate_shift, 25);
        assert_eq!(config.attention_q_learning_rate_shift, 20);
        assert_eq!(config.attention_qk_learning_rate_shift, 21);
        assert_eq!(config.target_frequency_cap, 8);
        assert_eq!(config.target_frequency_min_weight_q15, 2048);
        assert_eq!(config.argmax_margin_weight_q15, 1024);
        assert_eq!(config.target_segment, TargetSegment::GeneratedText);
        assert_eq!(config.target_token_min, IMAGE_BASE);
        assert_eq!(config.target_token_max, IMAGE_BASE + IMAGE_BINS - 1);
        assert_eq!(config.text_token_profile, TextTokenProfile::Chunked);
        assert!(config.zero_output_head_init);
        assert!(config.solomon_name_copy_init);
        assert!(config.reject_loss_regression);
    }

    #[test]
    fn parses_body_first_after_he_target_and_preserve_repair() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--target-segment",
                "body-first-after-he",
                "--solomon-name-copy-repair-preserve-body-output",
                "--solomon-body-scaffold",
                "--solomon-body-opening-repair",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.target_segment, TargetSegment::BodyFirstAfterHe);
        assert!(config.solomon_name_copy_repair);
        assert!(config.solomon_name_copy_repair_preserve_body_output);
        assert!(config.solomon_body_scaffold);
        assert!(config.solomon_body_opening_repair);
    }

    #[test]
    fn solomon_name_copy_init_samples_prompt_bound_opening() {
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        apply_solomon_name_copy_init(&mut transformer, TextTokenProfile::Chunked).unwrap();
        let model = SolomonAttentionModel {
            token_count: 0,
            token_hash: 0,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: None,
            transformer,
        };

        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: 4,
                min_text_tokens: 4,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 13,
            },
            None,
            None,
            None,
            NAME_OPENING_PREFIX,
        )
        .unwrap();

        assert_eq!(sample.text, "Solomon selects Bael: He");
    }

    #[test]
    fn solomon_name_copy_repair_restores_opening_over_bad_head_bias() {
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        transformer.up_weights.fill(0);
        transformer.gate_weights.fill(0);
        transformer.down_weights.fill(0);
        transformer.output_weights.fill(0);
        transformer.output_weights[usize::from(TEXT_BASE + b'H') * MINI_TRANSFORMER_D_MODEL + 15] =
            64;
        repair_solomon_name_copy_scaffold(&mut transformer, TextTokenProfile::Chunked).unwrap();
        let model = SolomonAttentionModel {
            token_count: 0,
            token_hash: 0,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: None,
            transformer,
        };

        let sample = sample_model(
            &model,
            "seal of Stolas",
            SampleOptions {
                max_text_tokens: 4,
                min_text_tokens: 4,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 13,
            },
            None,
            None,
            None,
            NAME_OPENING_PREFIX,
        )
        .unwrap();

        assert_eq!(sample.text, "Solomon selects Stolas: He");
    }

    #[test]
    fn solomon_name_copy_repair_preserves_unowned_attention_rows() {
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
        let q_index = (head_dim + 3) * MINI_TRANSFORMER_D_MODEL + 7;
        let k_index = (head_dim + 4) * MINI_TRANSFORMER_D_MODEL + 8;
        let embedding_index = usize::from(TEXT_BASE + b'x') * MINI_TRANSFORMER_D_MODEL + head_dim;
        let position_index = head_dim;
        let output_index = usize::from(TEXT_BASE + b'y') * MINI_TRANSFORMER_D_MODEL + head_dim;
        let owned_output_index = usize::from(TEXT_BASE + b'z') * MINI_TRANSFORMER_D_MODEL;
        let copied_output_index =
            usize::from(TEXT_BASE + b'z') * MINI_TRANSFORMER_D_MODEL + head_dim + 2;
        transformer.q_weights[q_index] = 42;
        transformer.k_weights[k_index] = -17;
        transformer.embeddings[embedding_index] = 1234;
        transformer.position_embeddings[position_index] = -4321;
        transformer.output_weights[output_index] = 77;
        transformer.output_weights[owned_output_index] = 66;
        transformer.output_weights[copied_output_index] = 88;

        repair_solomon_name_copy_scaffold(&mut transformer, TextTokenProfile::Chunked).unwrap();

        assert_eq!(transformer.q_weights[q_index], 42);
        assert_eq!(transformer.k_weights[k_index], -17);
        assert_eq!(transformer.embeddings[embedding_index], 1234);
        assert_eq!(transformer.position_embeddings[position_index], -4321);
        assert_eq!(transformer.output_weights[output_index], 77);
        assert_eq!(transformer.output_weights[owned_output_index], 0);
        assert_eq!(transformer.output_weights[copied_output_index], 0);
        assert_eq!(transformer.q_weights[0], 8);
        assert_eq!(transformer.k_weights[1], 8);
    }

    #[test]
    fn solomon_name_copy_repair_can_preserve_body_output_rows() {
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        let body_token = usize::from(TEXT_CHUNK_BASE) + 3;
        let name_token = usize::from(TEXT_CHUNK_BASE) + TEXT_CHUNK_NAME_START;
        let colon_token = usize::from(TEXT_CHUNK_BASE) + 1;
        let body_owned_index = body_token * MINI_TRANSFORMER_D_MODEL;
        let body_copied_index =
            body_token * MINI_TRANSFORMER_D_MODEL + MINI_TRANSFORMER_D_MODEL / 2 + 2;
        let name_index = name_token * MINI_TRANSFORMER_D_MODEL;
        let colon_index = colon_token * MINI_TRANSFORMER_D_MODEL;
        transformer.output_weights[body_owned_index] = 41;
        transformer.output_weights[body_copied_index] = -42;
        transformer.output_weights[name_index] = 43;
        transformer.output_weights[colon_index] = 44;

        repair_solomon_name_copy_scaffold_with_options(
            &mut transformer,
            TextTokenProfile::Chunked,
            true,
        )
        .unwrap();

        assert_eq!(transformer.output_weights[body_owned_index], 41);
        assert_eq!(transformer.output_weights[body_copied_index], -42);
        assert_eq!(transformer.output_weights[name_index], 0);
        assert_eq!(transformer.output_weights[colon_index], 0);
    }

    #[test]
    fn solomon_body_scaffold_samples_clean_body_after_opening() {
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        apply_solomon_name_copy_init(&mut transformer, TextTokenProfile::Chunked).unwrap();
        apply_solomon_body_scaffold(&mut transformer, TextTokenProfile::Chunked).unwrap();
        let model = SolomonAttentionModel {
            token_count: 0,
            token_hash: 0,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: None,
            transformer,
        };

        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: 40,
                min_text_tokens: 24,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 13,
            },
            None,
            None,
            None,
            "Solomon selects Bael: He ",
        )
        .unwrap();

        assert_eq!(
            sample.text,
            "Solomon selects Bael: He is of the Goetia and teacheth with his ART in LINE."
        );
    }

    #[test]
    fn solomon_body_opening_repair_samples_source_specific_opening() {
        let mut examples = Vec::new();
        for &name in TEXT_CHUNKS.get(TEXT_CHUNK_NAME_START..).unwrap_or(&[]) {
            let opening = match name {
                "Vassago" => "This",
                "Alloces" => "His",
                "Murmur" => "and",
                _ => "He",
            };
            examples.push(TextMemoryExample {
                primary_name: String::from(name),
                prompt: format!("seal of {name}"),
                text_tokens: encode_text_tokens(
                    &format!("Solomon selects {name}: {opening} office."),
                    TextTokenProfile::Chunked,
                ),
                image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
            });
        }
        let memory = TextMemory {
            order: 32,
            examples,
        };
        let mut transformer = MiniTransformerMlpModel::new_initial_with_seq_len(32);
        apply_solomon_name_copy_init(&mut transformer, TextTokenProfile::Chunked).unwrap();
        repair_solomon_body_opening_scaffold(&mut transformer, TextTokenProfile::Chunked, &memory)
            .unwrap();
        let model = SolomonAttentionModel {
            token_count: 0,
            token_hash: 0,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: Some(memory),
            transformer,
        };

        for (prompt, expected) in [
            ("seal of Bael", "Solomon selects Bael: He"),
            ("seal of Vassago", "Solomon selects Vassago: This"),
            ("seal of Alloces", "Solomon selects Alloces: His"),
        ] {
            let sample = sample_model(
                &model,
                prompt,
                SampleOptions {
                    max_text_tokens: 4,
                    min_text_tokens: 4,
                    repeat_run_cap: 4,
                    no_repeat_ngram: 4,
                    conditioning_boost_q8: 0,
                    text_prior_boost_q8: 0,
                    text_prior_strict: false,
                    text_chunk_boost_q8: 0,
                    decode_logit_delta: false,
                    prompt_name_opening_prior: false,
                    suppress_name_chunks_after_opening: false,
                    top_k: 1,
                    sample_seed: 13,
                },
                None,
                None,
                None,
                NAME_OPENING_PREFIX,
            )
            .unwrap();

            assert_eq!(sample.text, expected);
        }
    }

    #[test]
    fn parses_name_opening_target_segment() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--target-segment",
                "name-opening",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.target_segment, TargetSegment::NameOpening);
    }

    #[test]
    fn parses_name_opening_tail_target_segment() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--target-segment",
                "name-opening-tail",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.target_segment, TargetSegment::NameOpeningTail);
    }

    #[test]
    fn parses_body_after_he_target_segment() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--target-segment",
                "body-after-he",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.target_segment, TargetSegment::BodyAfterHe);
    }

    #[test]
    fn parses_body_first_after_opening_target_segment() {
        let config = parse_args(
            [
                "train",
                "--tokens",
                "tokens.u8",
                "--model-out",
                "model.nsrllmm",
                "--target-segment",
                "body-first-after-opening",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.target_segment, TargetSegment::BodyFirstAfterOpening);
    }

    #[test]
    fn name_opening_start_sequence_matches_text_prefix_tokens() {
        let mut expected = vec![TEXT];
        expected.extend(
            NAME_OPENING_PREFIX
                .bytes()
                .map(|byte| TEXT_BASE.saturating_add(byte)),
        );

        assert_eq!(
            name_opening_start_sequence(TextTokenProfile::Char),
            expected
        );
        assert_eq!(expected.len(), 17);
        assert_eq!(
            name_opening_start_sequence(TextTokenProfile::Chunked),
            vec![TEXT, TEXT_CHUNK_BASE]
        );
        assert_eq!(
            name_opening_end_markers(TextTokenProfile::Char),
            vec![TEXT_BASE + b':' as u8, IMAGE, EOS]
        );
        assert_eq!(
            name_opening_end_markers(TextTokenProfile::Chunked),
            vec![TEXT_CHUNK_BASE + 1, IMAGE, EOS]
        );
        assert_eq!(
            body_first_after_opening_train_segment(TextTokenProfile::Chunked),
            MiniTransformerTargetSegment::first_after_sequence_before_any(
                &[TEXT_CHUNK_BASE + 1],
                &[IMAGE, EOS],
            )
            .unwrap()
        );
    }

    #[test]
    fn parses_sample_text_prefix_and_embedded_memory_switch() {
        let config = parse_args(
            [
                "sample",
                "--prompt",
                "seal of Bael",
                "--text-prefix",
                "Solomon selects ",
                "--conditioning-examples",
                "none",
                "--no-embedded-text-memory",
                "--embedded-text-lm-order",
                "4",
                "--text-prior-min-order",
                "2",
                "--text-chunk-boost-q8",
                "1024",
                "--decode-logit-delta",
                "--prompt-name-opening-prior",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.command, Command::Sample);
        assert_eq!(config.text_prefix, "Solomon selects ");
        assert_eq!(config.conditioning_examples, None);
        assert!(!config.use_embedded_text_memory);
        assert_eq!(config.embedded_text_lm_order, 4);
        assert_eq!(config.text_prior_min_order, 2);
        assert_eq!(config.text_chunk_boost_q8, 1024);
        assert!(config.decode_logit_delta);
        assert!(config.prompt_name_opening_prior);
    }

    #[test]
    fn text_prefix_seeds_generated_text() {
        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Char,
            text_memory: None,
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let prefix_tokens = encode_text_prefix_tokens("Solomon selects ", TextTokenProfile::Char);
        assert_eq!(prefix_tokens.last().copied(), Some(TEXT_BASE + b' '));
        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: prefix_tokens.len(),
                min_text_tokens: 0,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            None,
            None,
            "Solomon selects ",
        )
        .unwrap();

        assert_eq!(sample.text, "Solomon selects");
        assert_eq!(
            generated_text_context(&sample.generated_tokens)[..prefix_tokens.len()],
            prefix_tokens
        );
    }

    #[test]
    fn version_two_model_loads_without_embedded_text_memory() {
        let transformer = MiniTransformerMlpModel::new_initial_with_seq_len(8);
        let transformer_bytes = transformer.try_to_bytes().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MODEL_MAGIC);
        push_u32(&mut bytes, 2);
        push_u32(&mut bytes, u32::from(PAD));
        push_u32(&mut bytes, u32::try_from(VOCAB_SIZE).unwrap());
        push_u32(&mut bytes, u32::from(TEXT_BASE));
        push_u32(&mut bytes, u32::from(TEXT_COUNT));
        push_u32(&mut bytes, u32::from(IMAGE_BASE));
        push_u32(&mut bytes, u32::from(IMAGE_BINS));
        push_u32(&mut bytes, u32::try_from(SIGNATURE_GRID).unwrap());
        push_u32(
            &mut bytes,
            attention_kind_code(MiniTransformerAttentionKind::Base2Softmax),
        );
        push_u32(
            &mut bytes,
            position_policy_code(MiniTransformerPositionPolicy::LearnedAbsolute),
        );
        push_u32(
            &mut bytes,
            text_token_profile_code(TextTokenProfile::Chunked),
        );
        push_u32(
            &mut bytes,
            u32::try_from(transformer.context_seq_len).unwrap(),
        );
        push_u64(&mut bytes, 16);
        push_u64(&mut bytes, 7);
        push_u64(&mut bytes, transformer.model_hash());
        push_u64(&mut bytes, u64::try_from(transformer_bytes.len()).unwrap());
        bytes.extend_from_slice(&transformer_bytes);
        let hash = hash_bytes(&bytes);
        push_u64(&mut bytes, hash);

        let loaded = SolomonAttentionModel::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.token_count, 16);
        assert_eq!(loaded.token_hash, 7);
        assert_eq!(loaded.text_token_profile, TextTokenProfile::Chunked);
        assert_eq!(loaded.text_memory, None);
    }

    #[test]
    fn text_tokens_are_printable_ascii_only() {
        let tokens = encode_text_tokens("Bael\nSeal", TextTokenProfile::Char);
        assert!(tokens.iter().all(|&token| is_printable_text_token(token)));
        assert_eq!(
            decode_text_tokens(&tokens, TextTokenProfile::Char),
            "Bael Seal"
        );
    }

    #[test]
    fn normalize_text_removes_numeric_footnote_refs() {
        assert_eq!(
            normalize_text("Alloces: [25] His Office [note] remains."),
            "Alloces: His Office [note] remains."
        );
    }

    #[test]
    fn source_text_token_sanitizer_removes_numeric_footnote_refs() {
        let tokens = encode_normalized_text_tokens(
            "Solomon selects Alloces: [25] His Office is great.",
            TextTokenProfile::Chunked,
        );
        let sanitized = sanitize_source_text_tokens(&tokens, TextTokenProfile::Chunked);

        assert_eq!(
            decode_text_tokens(&sanitized, TextTokenProfile::Chunked),
            "Solomon selects Alloces: His Office is great."
        );
    }

    #[test]
    fn eval_token_diagnostics_reports_target_rank_and_margin() {
        let mut row = MiniTransformerNextTokenRow {
            logits_q8: [0_i32; VOCAB_SIZE],
            probabilities_q15: [0_i16; VOCAB_SIZE],
        };
        row.logits_q8[usize::from(TEXT_BASE + b'a')] = 100;
        row.logits_q8[usize::from(TEXT_BASE + b'b')] = 50;
        row.logits_q8[usize::from(TEXT_BASE + b'c')] = 40;

        let diagnostics = eval_token_diagnostics(
            &row,
            EvalPhase::Text,
            TextTokenProfile::Char,
            TEXT_BASE + b'b',
        );

        assert_eq!(diagnostics.predicted, TEXT_BASE + b'a');
        assert_eq!(diagnostics.target_rank, 2);
        assert_eq!(diagnostics.target_margin_q8, -50);
    }

    #[test]
    fn chunked_text_tokens_round_trip_common_solomon_phrase() {
        let tokens = encode_text_tokens(
            "Solomon selects Bael: He maketh thee Invisible.",
            TextTokenProfile::Chunked,
        );
        assert_eq!(tokens.first().copied(), Some(TEXT_CHUNK_BASE));
        assert!(tokens.iter().all(|&token| is_text_token(token)));
        assert!(tokens.len() < "Solomon selects Bael: He maketh thee Invisible.".len());
        assert_eq!(
            decode_text_tokens(&tokens, TextTokenProfile::Chunked),
            "Solomon selects Bael: He maketh thee Invisible."
        );
    }

    #[test]
    fn chunked_text_prefix_through_colon_keeps_prompt_specific_name_start() {
        let tokens = encode_text_tokens(
            "Solomon selects Bael: He maketh thee Invisible.",
            TextTokenProfile::Chunked,
        );
        let prefix = text_prefix_through_colon(&tokens);

        assert!(!prefix.is_empty());
        assert_eq!(
            decode_text_tokens(&prefix, TextTokenProfile::Chunked),
            "Solomon selects Bael:"
        );
    }

    #[test]
    fn prompt_name_opening_prior_uses_name_from_prompt() {
        let tokens = prompt_name_opening_tokens("seal of Stolas", TextTokenProfile::Chunked);
        assert_eq!(
            decode_text_tokens(&tokens, TextTokenProfile::Chunked),
            "Solomon selects Stolas: He"
        );
        let accented_tokens =
            prompt_name_opening_tokens("seal of Ronové", TextTokenProfile::Chunked);
        assert_eq!(
            decode_text_tokens(&accented_tokens, TextTokenProfile::Chunked),
            "Solomon selects Ronove: He"
        );
        let char_tokens = prompt_name_opening_tokens("seal of Bael", TextTokenProfile::Char);
        assert_eq!(
            decode_text_tokens(&char_tokens, TextTokenProfile::Char),
            "Solomon selects Bael: He"
        );

        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Chunked,
            text_memory: None,
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let sample = sample_model(
            &model,
            "seal of Stolas",
            SampleOptions {
                max_text_tokens: 4,
                min_text_tokens: 0,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: true,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            None,
            None,
            "",
        )
        .unwrap();

        assert_eq!(sample.text, "Solomon selects Stolas: He");

        let char_model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Char,
            text_memory: None,
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let char_opening_len =
            prompt_name_opening_tokens("seal of Bael", TextTokenProfile::Char).len();
        let char_sample = sample_model(
            &char_model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: char_opening_len,
                min_text_tokens: 0,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: true,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            None,
            None,
            "Solomon selects ",
        )
        .unwrap();

        assert_eq!(char_sample.text, "Solomon selects Bael: He");
    }

    #[test]
    fn text_chunk_table_fits_reserved_byte_vocab() {
        assert!(TEXT_CHUNKS.len() <= VOCAB_SIZE - usize::from(TEXT_CHUNK_BASE));
    }

    #[test]
    fn corpus_validation_accepts_full_byte_vocab() {
        assert!(validate_corpus(&[BOS, PROMPT, TEXT, IMAGE, EOS]).is_ok());
        assert!(validate_corpus(&[u8::MAX]).is_ok());
    }

    #[test]
    fn allowed_text_tokens_can_stop_only_after_minimum_text() {
        assert!(!allowed_tokens(Phase::Text, false, TextTokenProfile::Char).contains(&IMAGE));
        assert!(allowed_tokens(Phase::Text, true, TextTokenProfile::Char).contains(&IMAGE));
        assert!(
            allowed_tokens(Phase::Text, false, TextTokenProfile::Chunked)
                .contains(&TEXT_CHUNK_BASE)
        );
        assert!(
            allowed_tokens(Phase::Image, false, TextTokenProfile::Char)
                .iter()
                .all(|&token| is_image_token(token))
        );
    }

    #[test]
    fn repeat_run_cap_removes_last_repeated_text_token() {
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'a',
                logit_q8: 9,
                probability_q15: 9,
            },
            Candidate {
                token: TEXT_BASE + b'b',
                logit_q8: 1,
                probability_q15: 1,
            },
        ];
        apply_repeat_run_cap(&mut candidates, &[TEXT_BASE + b'a', TEXT_BASE + b'a'], 2);
        assert_eq!(
            candidates,
            vec![Candidate {
                token: TEXT_BASE + b'b',
                logit_q8: 1,
                probability_q15: 1,
            }]
        );
    }

    #[test]
    fn expected_token_boost_can_override_argmax() {
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'a',
                logit_q8: 100,
                probability_q15: 100,
            },
            Candidate {
                token: TEXT_BASE + b'b',
                logit_q8: 1,
                probability_q15: 1,
            },
        ];
        apply_expected_token_boost(&mut candidates, TEXT_BASE + b'b', 1_000);
        assert_eq!(
            choose_candidate(&mut candidates, 0, 0, 1).token,
            TEXT_BASE + b'b'
        );
    }

    #[test]
    fn no_repeat_ngram_removes_candidate_that_would_repeat_ngram() {
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'c',
                logit_q8: 9,
                probability_q15: 9,
            },
            Candidate {
                token: TEXT_BASE + b'd',
                logit_q8: 1,
                probability_q15: 1,
            },
        ];
        let history = [
            TEXT_BASE + b'a',
            TEXT_BASE + b'b',
            TEXT_BASE + b'c',
            TEXT_BASE + b'a',
            TEXT_BASE + b'b',
        ];
        apply_no_repeat_ngram(&mut candidates, &history, 3);
        assert_eq!(
            candidates,
            vec![Candidate {
                token: TEXT_BASE + b'd',
                logit_q8: 1,
                probability_q15: 1,
            }]
        );
    }

    #[test]
    fn padded_context_left_pads_with_training_pad_token() {
        let mut context = PaddedContext::new(5);
        assert_eq!(
            context.window(&[BOS, PROMPT], 5),
            &[PAD, PAD, PAD, BOS, PROMPT]
        );
        assert_eq!(
            context.window(&[PAD, BOS, PROMPT, TEXT, TEXT_BASE + b'S', EOS], 5),
            &[BOS, PROMPT, TEXT, TEXT_BASE + b'S', EOS]
        );
    }

    #[test]
    fn generated_text_context_starts_after_text_marker() {
        let history = [
            BOS,
            PROMPT,
            TEXT_BASE + b's',
            TEXT_BASE + b'e',
            TEXT_BASE + b'a',
            TEXT_BASE + b'l',
            TEXT,
            TEXT_BASE + b'S',
            TEXT_BASE + b'o',
        ];
        assert_eq!(
            generated_text_context(&history),
            vec![TEXT_BASE + b'S', TEXT_BASE + b'o']
        );
    }

    #[test]
    fn generated_text_after_opening_detects_chunked_colon_he_marker() {
        let name_token = TEXT_CHUNK_BASE.saturating_add(TEXT_CHUNK_NAME_START as u8);
        let before_opening = [BOS, PROMPT, TEXT, TEXT_CHUNK_BASE, name_token];
        assert!(!generated_text_is_after_opening(&before_opening));

        let after_opening = [
            BOS,
            PROMPT,
            TEXT,
            TEXT_CHUNK_BASE,
            name_token,
            TEXT_CHUNK_BASE + 1,
            TEXT_CHUNK_BASE + 2,
        ];
        assert!(generated_text_is_after_opening(&after_opening));

        let char_opening = [
            BOS,
            PROMPT,
            TEXT,
            TEXT_BASE + b':',
            TEXT_BASE + b' ',
            TEXT_BASE + b'H',
            TEXT_BASE + b'e',
            TEXT_BASE + b' ',
        ];
        assert!(!generated_text_is_after_opening(&char_opening));
    }

    #[test]
    fn suppress_name_chunk_candidates_keeps_non_name_text_tokens() {
        let name_token = TEXT_CHUNK_BASE.saturating_add(TEXT_CHUNK_NAME_START as u8);
        let mut candidates = vec![
            Candidate {
                token: name_token,
                logit_q8: 100,
                probability_q15: 100,
            },
            Candidate {
                token: TEXT_CHUNK_BASE + 2,
                logit_q8: 90,
                probability_q15: 90,
            },
            Candidate {
                token: TEXT_BASE + b'm',
                logit_q8: 80,
                probability_q15: 80,
            },
        ];

        suppress_name_chunk_candidates(&mut candidates);

        assert!(
            !candidates
                .iter()
                .any(|candidate| candidate.token == name_token)
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.token == TEXT_CHUNK_BASE + 2)
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.token == TEXT_BASE + b'm')
        );
    }

    #[test]
    fn strict_text_prior_keeps_seen_continuations_only() {
        let mut prior = TextPrior {
            order: 2,
            min_order: 0,
            start_tokens: Vec::new(),
            prompt_starts: Vec::new(),
            transitions: HashMap::new(),
        };
        add_text_prior_sequence(
            &mut prior,
            &[TEXT_BASE + b'S', TEXT_BASE + b'o', TEXT_BASE + b'l'],
        );
        let history = [BOS, PROMPT, TEXT, TEXT_BASE + b'S', TEXT_BASE + b'o'];
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'l',
                logit_q8: 0,
                probability_q15: 0,
            },
            Candidate {
                token: TEXT_BASE + b'x',
                logit_q8: 999,
                probability_q15: 999,
            },
        ];
        let start_tokens = prior.start_tokens.clone();
        apply_text_prior(
            &mut candidates,
            &history,
            &prior,
            &start_tokens,
            1_000,
            true,
        );
        assert_eq!(
            candidates,
            vec![Candidate {
                token: TEXT_BASE + b'l',
                logit_q8: 1_000,
                probability_q15: 1_000,
            }]
        );
    }

    #[test]
    fn text_prior_min_order_skips_low_order_fallbacks() {
        let mut prior = TextPrior {
            order: 3,
            min_order: 2,
            start_tokens: Vec::new(),
            prompt_starts: Vec::new(),
            transitions: HashMap::new(),
        };
        let mut low_counts = [0_u32; VOCAB_LEN];
        low_counts[usize::from(TEXT_BASE + b'x')] = 1;
        prior.transitions.insert(vec![TEXT_BASE + b'a'], low_counts);

        let history = [TEXT, TEXT_BASE + b'a', TEXT_BASE + b'b'];
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'x',
                logit_q8: 999,
                probability_q15: 999,
            },
            Candidate {
                token: TEXT_BASE + b'y',
                logit_q8: 1,
                probability_q15: 1,
            },
        ];
        apply_text_prior(&mut candidates, &history, &prior, &[], 1_000, true);

        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn prompt_specific_text_prior_uses_matching_spirit_prefix() {
        let mut prior = TextPrior {
            order: 2,
            min_order: 0,
            start_tokens: Vec::new(),
            prompt_starts: Vec::new(),
            transitions: HashMap::new(),
        };
        let bael_text = encode_text_tokens(
            "Solomon selects Bael: He maketh thee invisible.",
            TextTokenProfile::Char,
        );
        let agares_text = encode_text_tokens(
            "Solomon selects Agares: He maketh them run.",
            TextTokenProfile::Char,
        );
        add_text_prior_sequence(&mut prior, &bael_text);
        add_text_prior_sequence(&mut prior, &agares_text);
        add_text_prior_prompt_start(&mut prior, "Bael", "seal of Bael", &bael_text);
        add_text_prior_prompt_start(&mut prior, "Agares", "seal of Agares", &agares_text);

        let start_tokens = prior.start_tokens_for_prompt("seal of Bael").to_vec();
        let mut history = vec![BOS, PROMPT, TEXT];
        history.extend(encode_text_tokens(
            "Solomon selects",
            TextTokenProfile::Char,
        ));
        history.push(TEXT_BASE + b' ');
        let next_index = generated_text_context(&history).len();
        assert_eq!(start_tokens[next_index], TEXT_BASE + b'B');
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'B',
                logit_q8: 0,
                probability_q15: 0,
            },
            Candidate {
                token: TEXT_BASE + b'A',
                logit_q8: 999,
                probability_q15: 999,
            },
        ];
        apply_text_prior(
            &mut candidates,
            &history,
            &prior,
            &start_tokens,
            1_000,
            true,
        );

        assert_eq!(
            decode_text_tokens(&start_tokens, TextTokenProfile::Char),
            "Solomon selects Bael:"
        );
        assert_eq!(
            candidates,
            vec![Candidate {
                token: TEXT_BASE + b'B',
                logit_q8: 1_000,
                probability_q15: 1_000,
            }]
        );
    }

    #[test]
    fn generic_prompt_keeps_common_text_prior_prefix() {
        let mut prior = TextPrior {
            order: 2,
            min_order: 0,
            start_tokens: Vec::new(),
            prompt_starts: Vec::new(),
            transitions: HashMap::new(),
        };
        let bael_text = encode_text_tokens(
            "Solomon selects Bael: He maketh thee invisible.",
            TextTokenProfile::Char,
        );
        add_text_prior_sequence(&mut prior, &bael_text);
        add_text_prior_prompt_start(&mut prior, "Bael", "king solomon seal", &bael_text);

        assert_eq!(
            prior.start_tokens_for_prompt("king solomon seal"),
            prior.start_tokens.as_slice()
        );
    }

    #[test]
    fn embedded_text_memory_builds_prompt_specific_prior() {
        let memory = TextMemory {
            order: 4,
            examples: vec![TextMemoryExample {
                primary_name: String::from("Bael"),
                prompt: String::from("seal of Bael"),
                text_tokens: encode_text_tokens(
                    "Solomon selects Bael: He maketh thee to go Invisible.",
                    TextTokenProfile::Char,
                ),
                image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
            }],
        };
        let prior = TextPrior::from(&memory, TextTokenProfile::Char);

        assert_eq!(prior.order, 4);
        assert_eq!(prior.prompt_starts.len(), 2);
        assert_eq!(
            decode_text_tokens(
                prior.start_tokens_for_prompt("seal of Bael"),
                TextTokenProfile::Char
            ),
            "Solomon selects Bael:"
        );
        assert!(prior.transitions.contains_key(&vec![TEXT]));
    }

    #[test]
    fn embedded_text_lm_uses_requested_low_order_with_prompt_starts() {
        let memory = TextMemory {
            order: 32,
            examples: vec![TextMemoryExample {
                primary_name: String::from("Bael"),
                prompt: String::from("seal of Bael"),
                text_tokens: encode_text_tokens(
                    "Solomon selects Bael: He maketh thee to go Invisible.",
                    TextTokenProfile::Char,
                ),
                image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
            }],
        };
        let prior = TextPrior::from_memory_with_order(&memory, 3, 0, true, TextTokenProfile::Char);

        assert_eq!(prior.order, 3);
        assert_eq!(
            decode_text_tokens(
                prior.start_tokens_for_prompt("seal of Bael"),
                TextTokenProfile::Char
            ),
            "Solomon selects Bael:"
        );
        assert!(prior.transitions.contains_key(&vec![
            TEXT_BASE + b'a',
            TEXT_BASE + b'e',
            TEXT_BASE + b'l'
        ]));
        assert!(!prior.transitions.contains_key(&encode_text_tokens(
            "Solomon selects Bael",
            TextTokenProfile::Char
        )));
    }

    #[test]
    fn embedded_text_lm_can_scope_transitions_to_prompt() {
        let memory = TextMemory {
            order: 32,
            examples: vec![
                TextMemoryExample {
                    primary_name: String::from("Bael"),
                    prompt: String::from("seal of Bael"),
                    text_tokens: encode_text_tokens(
                        "Solomon selects Bael: He maketh thee visible.",
                        TextTokenProfile::Char,
                    ),
                    image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
                },
                TextMemoryExample {
                    primary_name: String::from("Agares"),
                    prompt: String::from("seal of Agares"),
                    text_tokens: encode_text_tokens(
                        "Solomon selects Agares: He maketh them run.",
                        TextTokenProfile::Char,
                    ),
                    image_tokens: vec![IMAGE_BASE + 1; SIGNATURE_BINS],
                },
            ],
        };
        let prior = TextPrior::from_memory_for_prompt_with_order(
            &memory,
            "seal of Bael",
            32,
            0,
            true,
            TextTokenProfile::Char,
        );
        let mut history = vec![TEXT];
        history.extend(encode_text_tokens(
            "Solomon selects Bael: He maketh the",
            TextTokenProfile::Char,
        ));
        let mut candidates = vec![
            Candidate {
                token: TEXT_BASE + b'e',
                logit_q8: 1,
                probability_q15: 1,
            },
            Candidate {
                token: TEXT_BASE + b'm',
                logit_q8: 999,
                probability_q15: 999,
            },
        ];

        apply_text_prior(&mut candidates, &history, &prior, &[], 1_000, true);

        assert_eq!(
            candidates,
            vec![Candidate {
                token: TEXT_BASE + b'e',
                logit_q8: 1257,
                probability_q15: 1257,
            }]
        );
    }

    #[test]
    fn strict_text_prior_match_survives_repeat_filters() {
        let text = "Solomon selects Bael: ha ha.";
        let memory = TextMemory {
            order: 32,
            examples: vec![TextMemoryExample {
                primary_name: String::from("Bael"),
                prompt: String::from("seal of Bael"),
                text_tokens: encode_text_tokens(text, TextTokenProfile::Char),
                image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
            }],
        };
        let prior = TextPrior::from_memory_for_prompt_with_order(
            &memory,
            "seal of Bael",
            32,
            0,
            true,
            TextTokenProfile::Char,
        );
        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Char,
            text_memory: None,
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: encode_text_tokens(text, TextTokenProfile::Char).len(),
                min_text_tokens: 0,
                repeat_run_cap: 4,
                no_repeat_ngram: 2,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8,
                text_prior_strict: true,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            Some(&prior),
            None,
            "",
        )
        .unwrap();

        assert_eq!(sample.text, text);
    }

    #[test]
    fn image_prior_tokens_override_image_generation() {
        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Char,
            text_memory: None,
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(8),
        };
        let image_prior_tokens = (0..SIGNATURE_BINS)
            .map(|index| IMAGE_BASE + u8::try_from(index % usize::from(IMAGE_BINS)).unwrap())
            .collect::<Vec<_>>();
        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: 0,
                min_text_tokens: 0,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8,
                text_prior_boost_q8: 0,
                text_prior_strict: false,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            None,
            Some(&image_prior_tokens),
            "",
        )
        .unwrap();

        assert_eq!(
            sample.image_bins.to_vec(),
            image_prior_tokens
                .iter()
                .map(|token| token - IMAGE_BASE)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn text_repeat_filters_ignore_prompt_tokens() {
        let text = "Solomon selects Bael: He maketh thee to go Invisible.";
        let text_tokens = encode_text_tokens(text, TextTokenProfile::Char);
        let memory = TextMemory {
            order: 8,
            examples: vec![TextMemoryExample {
                primary_name: String::from("Bael"),
                prompt: String::from("seal of Bael"),
                text_tokens,
                image_tokens: vec![IMAGE_BASE; SIGNATURE_BINS],
            }],
        };
        let prior = TextPrior::from(&memory, TextTokenProfile::Char);
        let model = SolomonAttentionModel {
            token_count: 16,
            token_hash: 7,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            text_token_profile: TextTokenProfile::Char,
            text_memory: Some(memory),
            transformer: MiniTransformerMlpModel::new_initial_with_seq_len(32),
        };

        let sample = sample_model(
            &model,
            "seal of Bael",
            SampleOptions {
                max_text_tokens: 64,
                min_text_tokens: 16,
                repeat_run_cap: 4,
                no_repeat_ngram: 4,
                conditioning_boost_q8: 0,
                text_prior_boost_q8: DEFAULT_EMBEDDED_TEXT_MEMORY_BOOST_Q8,
                text_prior_strict: true,
                text_chunk_boost_q8: 0,
                decode_logit_delta: false,
                prompt_name_opening_prior: false,
                suppress_name_chunks_after_opening: false,
                top_k: 1,
                sample_seed: 1,
            },
            None,
            Some(&prior),
            None,
            "",
        )
        .unwrap();

        assert!(sample.text.starts_with("Solomon selects Bael: "));
        assert!(!sample.text.starts_with("Solomon selects Baal: "));
    }
}
