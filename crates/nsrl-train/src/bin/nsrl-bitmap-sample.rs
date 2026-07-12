#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "nsrl.bitmap_sampler_trace.v1";
const MODEL_MAGIC_CV3: &[u8; 8] = b"NSRLCV3\n";
const MODEL_MAGIC_MCH: &[u8; 8] = b"NSRLMCH\n";
const MODEL_MAGIC_TCH: &[u8; 8] = b"NSRLTCH\n";
const LATENT_MODEL_MAGIC: &[u8; 8] = b"NSRLLAT1";
const KERNEL: usize = 9;
const HIDDEN_CHANNELS: usize = 8;
const POSITION_FEATURE_CHANNELS: usize = 6;
const TEXT_FEATURE_CHANNELS: usize = 16;
const SIGNATURE_GRID: usize = 16;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
const LAYOUT_INK_FLOOR: u16 = 32;
const LAYOUT_INK_MIDPOINT: u16 = 54;
const LAYOUT_INK_CEILING: u16 = 96;
const QUALITY_RING_BUCKETS: usize = 32;
const QUALITY_BUCKET_STEPS: i64 = 4;
const INK_THRESHOLD: u8 = 64;
const STRONG_INK_THRESHOLD: u8 = 160;
const SPECK_INK_THRESHOLD: u8 = 96;
const QUALITY_EDGE_CLEAR: usize = 3;
const HIDDEN_KERNELS: [[i8; KERNEL]; HIDDEN_CHANNELS] = [
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    [-2, -1, 0, -1, 0, 1, 0, 1, 2],
    [0, 1, 2, -1, 0, 1, -2, -1, 0],
    [-1, 2, -1, 2, 0, 2, -1, 2, -1],
];
const CORRUPTION_KINDS: [&str; 10] = [
    "pixel-dropout",
    "salt-pepper",
    "block-mask",
    "stroke-thin",
    "stroke-thicken",
    "line-drop",
    "mixed-noise",
    "coarse-erase",
    "box-blur",
    "noise-seed",
];

#[derive(Debug, Clone)]
struct Config {
    model_path: PathBuf,
    out_dir: PathBuf,
    sample_count: usize,
    candidate_multiplier: usize,
    diversity_weight: i64,
    border_clear: usize,
    passes: usize,
    preview_columns: usize,
    latent_model_path: Option<PathBuf>,
    attention_plan_path: Option<PathBuf>,
    prompt: Option<String>,
    text_weight: i64,
    seed: String,
    init_mode: InitMode,
    worker_count: usize,
}

impl Default for Config {
    fn default() -> Self {
        let out_dir = PathBuf::from(
            "data/processed/key-solomon-goetia-denoise-v1/baseline-three-layer-conv/samples",
        );
        Self {
            model_path: PathBuf::from(
                "data/processed/key-solomon-goetia-denoise-v1/baseline-three-layer-conv/model.nsrlcv3",
            ),
            out_dir,
            sample_count: 64,
            candidate_multiplier: 1,
            diversity_weight: 0,
            border_clear: 0,
            passes: 8,
            preview_columns: 8,
            latent_model_path: None,
            attention_plan_path: None,
            prompt: None,
            text_weight: 96,
            seed: "solomon-sampler-v1".to_string(),
            init_mode: InitMode::Noise,
            worker_count: default_worker_count(),
        }
    }
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|workers| workers.get())
        .unwrap_or(1)
        .max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitMode {
    Noise,
}

impl InitMode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "noise" => Ok(Self::Noise),
            _ => Err(format!("unknown init mode: {value}; expected noise").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Noise => "noise",
        }
    }
}

#[derive(Debug)]
struct ConvLayer {
    weights: [i8; KERNEL],
    condition_biases: Vec<i16>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct LayeredConvModel {
    image_size: usize,
    timesteps: usize,
    output_shift: u8,
    corruption_count: usize,
    layers: Vec<ConvLayer>,
}

#[derive(Debug)]
struct MultichannelLayer {
    condition_weights: Vec<Vec<i8>>,
    condition_biases: Vec<i16>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct MultichannelModel {
    image_size: usize,
    timesteps: usize,
    hidden_shift: u8,
    output_shift: u8,
    hidden_channels: usize,
    text_conditioned: bool,
    corruption_count: usize,
    layers: Vec<MultichannelLayer>,
}

#[derive(Debug)]
enum SampleModel {
    Conv3(LayeredConvModel),
    Multichannel(MultichannelModel),
}

impl SampleModel {
    fn format_name(&self) -> &'static str {
        match self {
            Self::Conv3(_) => "NSRLCV3",
            Self::Multichannel(model) => {
                if model.text_conditioned {
                    "NSRLTCH"
                } else {
                    "NSRLMCH"
                }
            }
        }
    }

    fn image_size(&self) -> usize {
        match self {
            Self::Conv3(model) => model.image_size,
            Self::Multichannel(model) => model.image_size,
        }
    }

    fn timesteps(&self) -> usize {
        match self {
            Self::Conv3(model) => model.timesteps,
            Self::Multichannel(model) => model.timesteps,
        }
    }

    fn corruption_count(&self) -> usize {
        match self {
            Self::Conv3(model) => model.corruption_count,
            Self::Multichannel(model) => model.corruption_count,
        }
    }

    fn layer_count(&self) -> usize {
        match self {
            Self::Conv3(model) => model.layers.len(),
            Self::Multichannel(model) => model.layers.len(),
        }
    }

    fn feature_channels(&self) -> usize {
        match self {
            Self::Conv3(_) => KERNEL,
            Self::Multichannel(model) => model.hidden_channels,
        }
    }

    fn text_conditioned(&self) -> bool {
        matches!(self, Self::Multichannel(model) if model.text_conditioned)
    }
}

#[derive(Debug)]
struct Cursor {
    bytes: Vec<u8>,
    offset: usize,
}

#[derive(Debug)]
struct Rng {
    state: u32,
}

#[derive(Debug, Clone)]
struct Candidate {
    source_index: usize,
    score: i64,
    text_distance: i64,
    quality: SampleQuality,
    signature: [u16; SIGNATURE_BINS],
    image: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct SignatureStats {
    global_mean: i16,
    row_means: [i16; SIGNATURE_GRID],
    col_means: [i16; SIGNATURE_GRID],
}

#[derive(Debug)]
struct FeatureCache {
    position_features: Vec<[i16; POSITION_FEATURE_CHANNELS]>,
    text_features: Option<TextFeatureCache>,
}

#[derive(Debug)]
struct TextFeatureCache {
    centers: Vec<i16>,
    global_mean: i16,
    static_features: Vec<[i16; TEXT_FEATURE_CHANNELS]>,
}

#[derive(Debug)]
struct LatentCondition {
    model_path: PathBuf,
    target_plan_path: PathBuf,
    prompt: String,
    latent_dim: usize,
    text_feature_count: usize,
    target_source: String,
    target_number: usize,
    target_name: String,
    target_score: i64,
    target_latent_score: i64,
    target_lexical_score: i64,
    target_signature: [u16; SIGNATURE_BINS],
}

#[derive(Debug)]
struct LatentTextModel {
    latent_dim: usize,
    text_feature_count: usize,
    text_encoder_shift: u8,
    decoder_shift: u8,
    text_weights: Vec<i8>,
    text_biases: Vec<i16>,
    decoder_weights: Vec<i8>,
    decoder_biases: [i16; SIGNATURE_BINS],
}

#[derive(Debug)]
struct LatentPromptPrediction {
    source: String,
    number: usize,
    name: String,
    score: i64,
    signature: [u16; SIGNATURE_BINS],
}

#[derive(Debug, Clone, Copy)]
struct SampleQuality {
    total_score: i64,
    ring_score: i64,
    ring_coverage_score: i64,
    ring_balance_penalty: i64,
    stroke_score: i64,
    interior_score: i64,
    density_penalty: i64,
    border_penalty: i64,
    outside_penalty: i64,
    speck_penalty: i64,
    wash_penalty: i64,
}

#[derive(Debug)]
struct ScoreSummary {
    candidate_count: usize,
    selected_count: usize,
    diversity_weight: i64,
    selected_min_score: i64,
    selected_max_score: i64,
    selected_mean_score_q8: i64,
    selected_min_text_distance: i64,
    selected_mean_text_distance_q8: i64,
    selected_min_signature_distance: i64,
    selected_mean_signature_distance_q8: i64,
    selected_mean_ring_score_q8: i64,
    selected_mean_ring_coverage_score_q8: i64,
    selected_mean_ring_balance_penalty_q8: i64,
    selected_mean_stroke_score_q8: i64,
    selected_mean_interior_score_q8: i64,
    selected_mean_density_penalty_q8: i64,
    selected_mean_border_penalty_q8: i64,
    selected_mean_outside_penalty_q8: i64,
    selected_mean_speck_penalty_q8: i64,
    selected_mean_wash_penalty_q8: i64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-bitmap-sample: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.sample_count == 0
        || config.candidate_multiplier == 0
        || config.passes == 0
        || config.preview_columns == 0
    {
        return Err(
            "samples, candidate multiplier, passes, and preview columns must be positive".into(),
        );
    }
    if config.diversity_weight < 0 {
        return Err("--diversity-weight must be non-negative".into());
    }
    if config.text_weight < 0 {
        return Err("--text-weight must be non-negative".into());
    }
    if config.latent_model_path.is_some() && config.attention_plan_path.is_some() {
        return Err("--latent-model and --attention-plan are mutually exclusive".into());
    }
    if config.latent_model_path.is_some() && config.attention_plan_path.is_some() {
        return Err("--latent-model and --attention-plan are mutually exclusive".into());
    }
    if config.latent_model_path.is_some() && config.prompt.is_none() {
        return Err("--latent-model requires --prompt".into());
    }
    if config.prompt.is_some()
        && config.latent_model_path.is_none()
        && config.attention_plan_path.is_none()
    {
        return Err("--prompt requires --latent-model or --attention-plan".into());
    }
    fs::create_dir_all(&config.out_dir)?;

    let model = read_model(&config.model_path)?;
    let conditions = generation_conditions(&model, config.init_mode);
    if conditions.is_empty() {
        return Err("model has no active blend conditions".into());
    }

    let image_size = model.image_size();
    let image_bytes = checked_image_bytes(image_size)?;
    let latent_condition = if config.attention_plan_path.is_some() {
        Some(read_attention_plan_condition(&config)?)
    } else if config.latent_model_path.is_some() {
        Some(read_latent_condition(&config)?)
    } else {
        None
    };
    let output_count = config.sample_count;
    if model.text_conditioned() && latent_condition.is_none() {
        return Err("NSRLTCH text-conditioned models require --latent-model with --prompt".into());
    }
    let candidate_count = output_count
        .checked_mul(config.candidate_multiplier)
        .ok_or("candidate count overflow")?;
    let target_signature = active_target_signature(latent_condition.as_ref());
    let feature_cache = build_feature_cache(&model, image_size, target_signature)?;
    let candidates = sample_candidates(
        &config,
        &model,
        &conditions,
        image_size,
        candidate_count,
        target_signature,
        feature_cache.as_ref(),
    )?;
    let selected = select_candidates(candidates, output_count, config.diversity_weight)?;
    let score_summary = summarize_selection(&selected, candidate_count, config.diversity_weight)?;

    let mut samples = Vec::with_capacity(output_count * image_bytes);
    let mut preview = Vec::new();
    for candidate in &selected {
        samples.extend_from_slice(&candidate.image);
        preview.push(candidate.image.clone());
    }

    let raw_path = config.out_dir.join(format!("samples.ink{}.u8", image_size));
    fs::write(&raw_path, &samples)?;
    let sheet = make_contact_sheet(&preview, image_size, config.preview_columns, 4);
    let pgm_path = config.out_dir.join("samples.pgm");
    write_pgm(&pgm_path, sheet.width, sheet.height, &sheet.bytes)?;
    write_trace(
        &config,
        &model,
        &conditions,
        &raw_path,
        &pgm_path,
        latent_condition.as_ref(),
        &score_summary,
    )?;

    println!(
        "{{\"schema\":\"{}\",\"samples\":{},\"passes\":{},\"init_mode\":\"{}\",\"out_dir\":\"{}\",\"pgm\":\"{}\"}}",
        SCHEMA,
        output_count,
        config.passes,
        config.init_mode.as_str(),
        json_escape(&config.out_dir.display().to_string()),
        json_escape(&pgm_path.display().to_string()),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-bitmap-sample [--model PATH] [--out-dir PATH] [--samples N] [--candidate-multiplier N] [--diversity-weight N] [--border-clear N] [--passes N] [--preview-columns N] [--latent-model PATH --prompt TEXT | --attention-plan PATH [--prompt TEXT]] [--text-weight N] [--seed TEXT] [--init noise] [--workers N]"
    );
}

fn parse_args<I>(mut args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let mut config = Config::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--model" => {
                config.model_path = PathBuf::from(args.next().ok_or("--model requires PATH")?);
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
            }
            "--samples" => {
                config.sample_count = args.next().ok_or("--samples requires N")?.parse()?;
            }
            "--candidate-multiplier" => {
                config.candidate_multiplier = args
                    .next()
                    .ok_or("--candidate-multiplier requires N")?
                    .parse()?;
            }
            "--diversity-weight" => {
                config.diversity_weight = args
                    .next()
                    .ok_or("--diversity-weight requires N")?
                    .parse()?;
            }
            "--border-clear" => {
                config.border_clear = args.next().ok_or("--border-clear requires N")?.parse()?;
            }
            "--passes" => {
                config.passes = args.next().ok_or("--passes requires N")?.parse()?;
            }
            "--preview-columns" => {
                config.preview_columns =
                    args.next().ok_or("--preview-columns requires N")?.parse()?;
            }
            "--text-index" => {
                let _ = args.next().ok_or("--text-index requires PATH")?;
                return Err(
                    "--text-index target retrieval has been removed; use --latent-model".into(),
                );
            }
            "--latent-model" => {
                config.latent_model_path = Some(PathBuf::from(
                    args.next().ok_or("--latent-model requires PATH")?,
                ));
            }
            "--attention-plan" => {
                config.attention_plan_path = Some(PathBuf::from(
                    args.next().ok_or("--attention-plan requires PATH")?,
                ));
            }
            "--prompt" => {
                config.prompt = Some(args.next().ok_or("--prompt requires TEXT")?);
            }
            "--text-all" => {
                return Err(
                    "--text-all target retrieval has been removed; use --latent-model --prompt"
                        .into(),
                );
            }
            "--text-weight" => {
                config.text_weight = args.next().ok_or("--text-weight requires N")?.parse()?;
            }
            "--seed" => {
                config.seed = args.next().ok_or("--seed requires TEXT")?;
            }
            "--init" => {
                config.init_mode = InitMode::parse(&args.next().ok_or("--init requires MODE")?)?;
            }
            "--workers" => {
                config.worker_count = args.next().ok_or("--workers requires N")?.parse()?;
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(config)
}

fn read_model(path: &Path) -> Result<SampleModel, Box<dyn std::error::Error>> {
    let mut cursor = Cursor {
        bytes: fs::read(path)?,
        offset: 0,
    };
    let magic = cursor.read_bytes(MODEL_MAGIC_CV3.len())?.to_vec();
    let model = if magic.as_slice() == MODEL_MAGIC_CV3 {
        SampleModel::Conv3(read_cv3_model(&mut cursor)?)
    } else if magic.as_slice() == MODEL_MAGIC_MCH {
        SampleModel::Multichannel(read_multichannel_model(&mut cursor, false)?)
    } else if magic.as_slice() == MODEL_MAGIC_TCH {
        SampleModel::Multichannel(read_multichannel_model(&mut cursor, true)?)
    } else {
        return Err(format!("{} is not a sampler model", path.display()).into());
    };
    if cursor.offset != cursor.bytes.len() {
        return Err(format!(
            "{} has {} trailing bytes",
            path.display(),
            cursor.bytes.len() - cursor.offset
        )
        .into());
    }
    Ok(model)
}

fn read_cv3_model(cursor: &mut Cursor) -> Result<LayeredConvModel, Box<dyn std::error::Error>> {
    let image_size = usize::try_from(cursor.read_u32()?)?;
    let timesteps = usize::try_from(cursor.read_u32()?)?;
    let output_shift = u8::try_from(cursor.read_u32()?)?;
    let corruption_count = usize::try_from(cursor.read_u32()?)?;
    let layer_count = usize::try_from(cursor.read_u32()?)?;
    let condition_count = corruption_count
        .checked_mul(timesteps)
        .ok_or("condition count overflow")?;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let mut weights = [0_i8; KERNEL];
        for weight in &mut weights {
            *weight = cursor.read_i8()?;
        }
        let mut condition_biases = Vec::with_capacity(condition_count);
        for _ in 0..condition_count {
            condition_biases.push(cursor.read_i16()?);
        }
        let condition_blend_shifts = cursor.read_bytes(condition_count)?.to_vec();
        layers.push(ConvLayer {
            weights,
            condition_biases,
            condition_blend_shifts,
        });
    }
    Ok(LayeredConvModel {
        image_size,
        timesteps,
        output_shift,
        corruption_count,
        layers,
    })
}

fn read_multichannel_model(
    cursor: &mut Cursor,
    text_conditioned: bool,
) -> Result<MultichannelModel, Box<dyn std::error::Error>> {
    let image_size = usize::try_from(cursor.read_u32()?)?;
    let timesteps = usize::try_from(cursor.read_u32()?)?;
    let hidden_shift = u8::try_from(cursor.read_u32()?)?;
    let output_shift = u8::try_from(cursor.read_u32()?)?;
    let hidden_channels = usize::try_from(cursor.read_u32()?)?;
    let max_channels = if text_conditioned {
        HIDDEN_KERNELS.len() + POSITION_FEATURE_CHANNELS + TEXT_FEATURE_CHANNELS
    } else {
        HIDDEN_KERNELS.len() + POSITION_FEATURE_CHANNELS
    };
    let min_channels = if text_conditioned {
        HIDDEN_KERNELS.len() + POSITION_FEATURE_CHANNELS + TEXT_FEATURE_CHANNELS
    } else {
        HIDDEN_KERNELS.len()
    };
    if hidden_channels == 0 || hidden_channels > max_channels {
        return Err(format!(
            "unsupported hidden channel count: {hidden_channels}; sampler supports {max_channels}"
        )
        .into());
    }
    if hidden_channels < min_channels {
        return Err(format!(
            "unsupported hidden channel count: {hidden_channels}; {} models require at least {min_channels} channels",
            if text_conditioned { "NSRLTCH" } else { "NSRLMCH" }
        )
        .into());
    }
    let corruption_count = usize::try_from(cursor.read_u32()?)?;
    let layer_count = usize::try_from(cursor.read_u32()?)?;
    let condition_count = corruption_count
        .checked_mul(timesteps)
        .ok_or("condition count overflow")?;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let mut condition_weights = Vec::with_capacity(condition_count);
        for _ in 0..condition_count {
            let mut weights = Vec::with_capacity(hidden_channels);
            for _ in 0..hidden_channels {
                weights.push(cursor.read_i8()?);
            }
            condition_weights.push(weights);
        }
        let mut condition_biases = Vec::with_capacity(condition_count);
        for _ in 0..condition_count {
            condition_biases.push(cursor.read_i16()?);
        }
        let condition_blend_shifts = cursor.read_bytes(condition_count)?.to_vec();
        layers.push(MultichannelLayer {
            condition_weights,
            condition_biases,
            condition_blend_shifts,
        });
    }
    Ok(MultichannelModel {
        image_size,
        timesteps,
        hidden_shift,
        output_shift,
        hidden_channels,
        text_conditioned,
        corruption_count,
        layers,
    })
}

impl Cursor {
    fn read_bytes(&mut self, count: usize) -> Result<&[u8], Box<dyn std::error::Error>> {
        let end = self.offset.checked_add(count).ok_or("cursor overflow")?;
        if end > self.bytes.len() {
            return Err("unexpected end of model".into());
        }
        let start = self.offset;
        self.offset = end;
        Ok(&self.bytes[start..end])
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i16(&mut self) -> Result<i16, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i8(&mut self) -> Result<i8, Box<dyn std::error::Error>> {
        Ok(self.read_bytes(1)?[0] as i8)
    }
}

fn read_latent_condition(config: &Config) -> Result<LatentCondition, Box<dyn std::error::Error>> {
    let model_path = config
        .latent_model_path
        .as_ref()
        .ok_or("--latent-model is required")?;
    let prompt = config.prompt.as_ref().ok_or("--prompt is required")?;
    let model = read_latent_model(model_path)?;
    let prediction = model.prediction_for_prompt(prompt)?;
    Ok(LatentCondition {
        model_path: model_path.clone(),
        target_plan_path: PathBuf::new(),
        prompt: prompt.clone(),
        latent_dim: model.latent_dim,
        text_feature_count: model.text_feature_count,
        target_source: prediction.source,
        target_number: prediction.number,
        target_name: prediction.name,
        target_score: prediction.score,
        target_latent_score: prediction.score,
        target_lexical_score: 0,
        target_signature: prediction.signature,
    })
}

fn read_attention_plan_condition(
    config: &Config,
) -> Result<LatentCondition, Box<dyn std::error::Error>> {
    let plan_path = config
        .attention_plan_path
        .as_ref()
        .ok_or("--attention-plan is required")?;
    let plan = fs::read(plan_path)?;
    if plan.len() != SIGNATURE_BINS {
        return Err(format!(
            "{} must contain exactly {} u8 bins",
            plan_path.display(),
            SIGNATURE_BINS
        )
        .into());
    }
    let mut target_signature = [0_u16; SIGNATURE_BINS];
    for (target, &value) in target_signature.iter_mut().zip(plan.iter()) {
        *target = u16::from(value);
    }
    let prompt = config
        .prompt
        .clone()
        .unwrap_or_else(|| plan_path.display().to_string());
    let target_name = config.prompt.clone().unwrap_or_else(|| {
        plan_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("attention-plan")
            .to_string()
    });
    Ok(LatentCondition {
        model_path: PathBuf::new(),
        target_plan_path: plan_path.clone(),
        prompt,
        latent_dim: 0,
        text_feature_count: 0,
        target_source: "attention-plan".to_string(),
        target_number: 0,
        target_name,
        target_score: 0,
        target_latent_score: 0,
        target_lexical_score: 0,
        target_signature,
    })
}

fn read_latent_model(path: &Path) -> Result<LatentTextModel, Box<dyn std::error::Error>> {
    let mut cursor = Cursor {
        bytes: fs::read(path)?,
        offset: 0,
    };
    let magic = cursor.read_bytes(LATENT_MODEL_MAGIC.len())?.to_vec();
    if magic.as_slice() != LATENT_MODEL_MAGIC {
        return Err(format!("{} is not an NSRLLAT1 latent model", path.display()).into());
    }
    let latent_dim = usize::try_from(cursor.read_u32()?)?;
    let text_feature_count = usize::try_from(cursor.read_u32()?)?;
    let signature_bins = usize::try_from(cursor.read_u32()?)?;
    let text_encoder_shift = u8::try_from(cursor.read_u32()?)?;
    let _image_encoder_shift = u8::try_from(cursor.read_u32()?)?;
    let decoder_shift = u8::try_from(cursor.read_u32()?)?;
    let signature_grid = usize::try_from(cursor.read_u32()?)?;
    if latent_dim == 0
        || text_feature_count == 0
        || signature_bins != SIGNATURE_BINS
        || signature_grid != SIGNATURE_GRID
    {
        return Err(format!("{} has incompatible latent dimensions", path.display()).into());
    }
    let text_weight_count = latent_dim
        .checked_mul(text_feature_count)
        .ok_or("latent text weight count overflow")?;
    let image_weight_count = latent_dim
        .checked_mul(SIGNATURE_BINS)
        .ok_or("latent image weight count overflow")?;
    let decoder_weight_count = SIGNATURE_BINS
        .checked_mul(latent_dim)
        .ok_or("latent decoder weight count overflow")?;

    let mut text_weights = Vec::with_capacity(text_weight_count);
    for _ in 0..text_weight_count {
        text_weights.push(cursor.read_i8()?);
    }
    let mut text_biases = Vec::with_capacity(latent_dim);
    for _ in 0..latent_dim {
        text_biases.push(cursor.read_i16()?);
    }
    for _ in 0..image_weight_count {
        cursor.read_i8()?;
    }
    for _ in 0..latent_dim {
        cursor.read_i16()?;
    }
    let mut decoder_weights = Vec::with_capacity(decoder_weight_count);
    for _ in 0..decoder_weight_count {
        decoder_weights.push(cursor.read_i8()?);
    }
    let mut decoder_biases = [0_i16; SIGNATURE_BINS];
    for bias in &mut decoder_biases {
        *bias = cursor.read_i16()?;
    }
    if cursor.offset != cursor.bytes.len() {
        return Err(format!(
            "{} has {} trailing bytes",
            path.display(),
            cursor.bytes.len() - cursor.offset
        )
        .into());
    }
    Ok(LatentTextModel {
        latent_dim,
        text_feature_count,
        text_encoder_shift,
        decoder_shift,
        text_weights,
        text_biases,
        decoder_weights,
        decoder_biases,
    })
}

impl LatentTextModel {
    fn prediction_for_prompt(
        &self,
        prompt: &str,
    ) -> Result<LatentPromptPrediction, Box<dyn std::error::Error>> {
        let features = latent_text_features(prompt, self.text_feature_count);
        let text_latent = self.encode_text(&features)?;
        Ok(LatentPromptPrediction {
            source: "decoded-latent".to_string(),
            number: 0,
            name: prompt.to_string(),
            score: 0,
            signature: sharpen_signature(&self.decode_signature(&text_latent)),
        })
    }

    fn encode_text(&self, features: &[i16]) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
        if features.len() != self.text_feature_count {
            return Err("latent text feature count mismatch".into());
        }
        let mut out = vec![0_i16; self.latent_dim];
        let active_features: Vec<(usize, i16)> = features
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| *value != 0)
            .collect();
        for (dim, slot) in out.iter_mut().enumerate() {
            let mut acc = 0_i64;
            for &(feature, value) in &active_features {
                let weight = self.text_weights[dim * self.text_feature_count + feature];
                acc = acc.saturating_add(i64::from(weight) * i64::from(value));
            }
            let value = signed_round_shift(acc, self.text_encoder_shift)
                .saturating_add(i64::from(self.text_biases[dim]));
            *slot = value.clamp(-511, 511) as i16;
        }
        Ok(out)
    }

    fn decode_signature(&self, latent: &[i16]) -> [u16; SIGNATURE_BINS] {
        let mut out = [0_u16; SIGNATURE_BINS];
        for (bin, out_value) in out.iter_mut().enumerate() {
            let mut acc = i64::from(self.decoder_biases[bin]) << self.decoder_shift;
            for (dim, &latent_value) in latent.iter().enumerate().take(self.latent_dim) {
                let weight = self.decoder_weights[bin * self.latent_dim + dim];
                acc = acc.saturating_add(i64::from(weight) * i64::from(latent_value));
            }
            let value = signed_round_shift(acc, self.decoder_shift).clamp(0, 255);
            *out_value = u16::try_from(value).unwrap_or(0);
        }
        out
    }
}

fn sharpen_signature(input: &[u16; SIGNATURE_BINS]) -> [u16; SIGNATURE_BINS] {
    let mut out = [0_u16; SIGNATURE_BINS];
    for (target, &value) in out.iter_mut().zip(input.iter()) {
        *target = sharpen_signature_value(value);
    }
    out
}

fn sharpen_signature_value(value: u16) -> u16 {
    if value <= LAYOUT_INK_FLOOR {
        return 0;
    }
    if value >= LAYOUT_INK_CEILING {
        return 255;
    }
    if value >= LAYOUT_INK_MIDPOINT { 255 } else { 0 }
}

fn active_target_signature(
    latent_condition: Option<&LatentCondition>,
) -> Option<&[u16; SIGNATURE_BINS]> {
    latent_condition.map(|condition| &condition.target_signature)
}

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() {
            current.push(char::from(byte.to_ascii_lowercase()));
        } else if !current.is_empty() {
            push_normalized_token(&mut tokens, &mut current);
        }
    }
    if !current.is_empty() {
        push_normalized_token(&mut tokens, &mut current);
    }
    tokens
}

fn push_normalized_token(tokens: &mut Vec<String>, current: &mut String) {
    let normalized = normalize_token(current);
    current.clear();
    if normalized.len() >= 2 {
        tokens.push(normalized);
    }
}

fn normalize_token(token: &str) -> String {
    match token {
        "teach" | "teacher" | "teaches" | "teacheth" | "teaching" => {
            return "teach".to_string();
        }
        "know" | "knows" | "known" | "knowing" | "knoweth" | "knowledge" => {
            return "know".to_string();
        }
        "make" | "makes" | "maketh" | "making" => return "make".to_string(),
        "discover" | "discovers" | "discovereth" | "discovering" => {
            return "discover".to_string();
        }
        "produce" | "produces" | "produceth" | "producing" => return "produce".to_string(),
        "answer" | "answers" | "answereth" | "answering" => return "answer".to_string(),
        "virtue" | "virtues" => return "virtue".to_string(),
        "water" | "waters" => return "water".to_string(),
        "rush" | "rushing" | "rushings" => return "rush".to_string(),
        "herb" | "herbs" => return "herb".to_string(),
        "stone" | "stones" => return "stone".to_string(),
        "science" | "sciences" => return "science".to_string(),
        _ => {}
    }
    if token.len() > 5 && (token.ends_with("eth") || token.ends_with("ing")) {
        token[..token.len() - 3].to_string()
    } else if token.len() > 4 && token.ends_with("es") {
        token[..token.len() - 2].to_string()
    } else if token.len() > 3 && token.ends_with('s') {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

fn latent_text_features(text: &str, feature_count: usize) -> Vec<i16> {
    let mut features = vec![0_i16; feature_count];
    let tokens = tokenize_text(text);
    if !tokens.is_empty() && tokens.len() <= 4 {
        add_latent_text_feature(&mut features, "whole", &tokens.join(" "), 0, 320);
    }
    for (position, token) in tokens.iter().enumerate() {
        add_latent_text_feature(&mut features, "tok", token, position, 72);
        if let Some(next) = tokens.get(position + 1) {
            add_latent_text_feature(
                &mut features,
                "bi",
                &format!("{token} {next}"),
                position,
                96,
            );
        }
        if let (Some(next), Some(third)) = (tokens.get(position + 1), tokens.get(position + 2)) {
            add_latent_text_feature(
                &mut features,
                "tri",
                &format!("{token} {next} {third}"),
                position,
                112,
            );
        }
    }
    let content = content_tokens(&tokens);
    if !content.is_empty() && content.len() <= 5 {
        add_latent_text_feature(&mut features, "cwhole", &content.join(" "), 0, 336);
        add_latent_text_feature(
            &mut features,
            "cset",
            &sorted_feature_key(&content.iter().map(String::as_str).collect::<Vec<_>>()),
            0,
            336,
        );
    }
    for (position, token) in content.iter().enumerate() {
        add_latent_text_feature(&mut features, "ctok", token, position, 128);
        if let Some(next) = content.get(position + 1) {
            add_latent_text_feature(
                &mut features,
                "cbi",
                &format!("{token} {next}"),
                position,
                160,
            );
        }
        if let (Some(next), Some(third)) = (content.get(position + 1), content.get(position + 2)) {
            add_latent_text_feature(
                &mut features,
                "ctri",
                &format!("{token} {next} {third}"),
                position,
                176,
            );
        }
        let window_end = (position + 16).min(content.len());
        for right in (position + 1)..window_end {
            add_latent_text_feature(
                &mut features,
                "skip2",
                &format!("{token} {}", content[right]),
                position,
                176,
            );
            add_latent_text_feature(
                &mut features,
                "pair",
                &sorted_feature_key(&[token.as_str(), content[right].as_str()]),
                position,
                192,
            );
            for third in (right + 1)..window_end {
                add_latent_text_feature(
                    &mut features,
                    "triple",
                    &sorted_feature_key(&[
                        token.as_str(),
                        content[right].as_str(),
                        content[third].as_str(),
                    ]),
                    position,
                    192,
                );
            }
        }
    }
    features
}

fn content_tokens(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|token| token.len() >= 3 && !is_stopword(token))
        .cloned()
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "about"
            | "after"
            | "again"
            | "all"
            | "also"
            | "an"
            | "and"
            | "any"
            | "are"
            | "as"
            | "at"
            | "be"
            | "before"
            | "both"
            | "but"
            | "by"
            | "can"
            | "etc"
            | "for"
            | "from"
            | "great"
            | "have"
            | "he"
            | "her"
            | "him"
            | "his"
            | "in"
            | "is"
            | "it"
            | "man"
            | "many"
            | "men"
            | "must"
            | "of"
            | "or"
            | "order"
            | "seal"
            | "shall"
            | "she"
            | "spirit"
            | "spirits"
            | "the"
            | "this"
            | "thou"
            | "to"
            | "unto"
            | "upon"
            | "which"
            | "who"
            | "will"
            | "with"
    )
}

fn sorted_feature_key(parts: &[&str]) -> String {
    let mut parts = parts.to_vec();
    parts.sort_unstable();
    parts.join("\u{0}")
}

fn add_latent_text_feature(
    features: &mut [i16],
    namespace: &str,
    text: &str,
    position: usize,
    base_value: i16,
) {
    if text.len() < 2 || features.is_empty() {
        return;
    }
    let hash = hash_seed(&[namespace, text]);
    let bin = usize::try_from(hash).unwrap_or(0) % features.len();
    let length = i16::try_from(text.len().min(28)).unwrap_or(28);
    let position_bonus = i16::try_from(position % 7).unwrap_or(0).saturating_mul(5);
    let value = base_value
        .saturating_add(length.saturating_mul(6))
        .saturating_add(position_bonus)
        .min(384);
    let signed = if hash & 0x8000_0000 == 0 {
        value
    } else {
        -value
    };
    features[bin] = features[bin].saturating_add(signed).clamp(-511, 511);
}

fn active_conditions(model: &SampleModel) -> Vec<usize> {
    let timesteps = model.timesteps();
    let condition_count = model.corruption_count() * timesteps;
    let mut conditions = Vec::new();
    for condition in 0..condition_count {
        if model_has_active_condition(model, condition) {
            conditions.push(condition);
        }
    }
    conditions.sort_by(|left, right| {
        let left_timestep = left % timesteps;
        let right_timestep = right % timesteps;
        right_timestep
            .cmp(&left_timestep)
            .then_with(|| left.cmp(right))
    });
    conditions
}

fn generation_conditions(model: &SampleModel, init_mode: InitMode) -> Vec<usize> {
    let conditions = active_conditions(model);
    if init_mode != InitMode::Noise {
        return conditions;
    }
    let Some(noise_kind) = CORRUPTION_KINDS
        .iter()
        .position(|&kind| kind == "noise-seed")
    else {
        return conditions;
    };
    if noise_kind >= model.corruption_count() {
        return conditions;
    }
    let timesteps = model.timesteps();
    let noise_conditions: Vec<usize> = conditions
        .iter()
        .copied()
        .filter(|condition| condition / timesteps == noise_kind)
        .collect();
    if noise_conditions.is_empty() {
        conditions
    } else {
        let mut scheduled = noise_conditions;
        scheduled.extend(
            conditions
                .iter()
                .copied()
                .filter(|condition| condition / timesteps != noise_kind),
        );
        scheduled
    }
}

fn model_has_active_condition(model: &SampleModel, condition: usize) -> bool {
    match model {
        SampleModel::Conv3(model) => model.layers.iter().any(|layer| {
            layer
                .condition_blend_shifts
                .get(condition)
                .copied()
                .unwrap_or(u8::MAX)
                != u8::MAX
        }),
        SampleModel::Multichannel(model) => model.layers.iter().any(|layer| {
            layer
                .condition_blend_shifts
                .get(condition)
                .copied()
                .unwrap_or(u8::MAX)
                != u8::MAX
        }),
    }
}

fn sample_candidates(
    config: &Config,
    model: &SampleModel,
    conditions: &[usize],
    image_size: usize,
    candidate_count: usize,
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
    feature_cache: Option<&FeatureCache>,
) -> Result<Vec<Candidate>, Box<dyn std::error::Error>> {
    let worker_count = config.worker_count.max(1).min(candidate_count.max(1));
    if worker_count == 1 || candidate_count <= 1 {
        let mut candidates = Vec::with_capacity(candidate_count);
        for candidate_index in 0..candidate_count {
            candidates.push(sample_candidate(
                config,
                model,
                conditions,
                image_size,
                candidate_index,
                target_signature,
                feature_cache,
            )?);
        }
        return Ok(candidates);
    }

    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let start = worker_index * candidate_count / worker_count;
            let end = (worker_index + 1) * candidate_count / worker_count;
            handles.push(scope.spawn(move || {
                let mut chunk = Vec::with_capacity(end.saturating_sub(start));
                for candidate_index in start..end {
                    let candidate = sample_candidate(
                        config,
                        model,
                        conditions,
                        image_size,
                        candidate_index,
                        target_signature,
                        feature_cache,
                    )
                    .map_err(|error| error.to_string());
                    chunk.push((candidate_index, candidate));
                }
                chunk
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join())
            .collect::<Result<Vec<_>, _>>()
    })
    .map_err(|_| "candidate worker panicked")?;

    let mut candidates = vec![None; candidate_count];
    for chunk in chunks {
        for (candidate_index, candidate) in chunk {
            candidates[candidate_index] =
                Some(candidate.map_err(|error| format!("candidate {candidate_index}: {error}"))?);
        }
    }
    candidates
        .into_iter()
        .enumerate()
        .map(|(candidate_index, candidate)| {
            candidate.ok_or_else(|| format!("candidate {candidate_index} was not sampled").into())
        })
        .collect()
}

fn sample_candidate(
    config: &Config,
    model: &SampleModel,
    conditions: &[usize],
    image_size: usize,
    candidate_index: usize,
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
    feature_cache: Option<&FeatureCache>,
) -> Result<Candidate, Box<dyn std::error::Error>> {
    let mut image = initial_image(config, image_size, candidate_index)?;
    for _ in 0..config.passes {
        for &condition in conditions {
            image =
                apply_model_condition(model, condition, &image, target_signature, feature_cache)?;
        }
    }
    clear_border(&mut image, image_size, config.border_clear);
    let quality = score_sample(&image, image_size)?;
    let signature = sample_signature(&image, image_size)?;
    let text_distance = target_signature
        .map(|target| text_signature_distance(&signature, target))
        .unwrap_or(0);
    let text_penalty = text_distance.saturating_mul(config.text_weight);
    let score = quality.total_score.saturating_sub(text_penalty);
    Ok(Candidate {
        source_index: candidate_index,
        score,
        text_distance,
        quality,
        signature,
        image,
    })
}

fn initial_image(
    config: &Config,
    image_size: usize,
    sample_index: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let seed = hash_seed(&[
        &config.seed,
        &sample_index.to_string(),
        config.init_mode.as_str(),
    ]);
    let mut rng = Rng::new(seed);
    match config.init_mode {
        InitMode::Noise => Ok(initial_noise(image_size, &mut rng)?),
    }
}

fn initial_noise(image_size: usize, rng: &mut Rng) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let mut image = vec![0_u8; image_bytes];
    for value in &mut image {
        let roll = rng.rand_bounded(16);
        *value = if roll == 0 {
            255
        } else if roll == 1 {
            160
        } else {
            0
        };
    }
    Ok(image)
}

fn transform_coords(image_size: usize, x: usize, y: usize, variant: usize) -> (usize, usize) {
    let last = image_size - 1;
    match variant % 8 {
        0 => (x, y),
        1 => (last - x, y),
        2 => (x, last - y),
        3 => (last - x, last - y),
        4 => (y, x),
        5 => (last - y, x),
        6 => (y, last - x),
        _ => (last - y, last - x),
    }
}

fn score_sample(
    image: &[u8],
    image_size: usize,
) -> Result<SampleQuality, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!("sample has {} bytes, expected {image_bytes}", image.len()).into());
    }
    let center = i64::try_from(image_size / 2)?;
    let outer_radius = i64::try_from(image_size / 2)?.saturating_sub(10);
    let inner_radius = outer_radius.saturating_sub(11);
    let outer_low = outer_radius.saturating_sub(5);
    let outer_high = outer_radius.saturating_add(5);
    let inner_low = inner_radius.saturating_sub(4);
    let inner_high = inner_radius.saturating_add(4);
    let center_radius = inner_radius.saturating_sub(7);
    let outer_low2 = outer_low.saturating_mul(outer_low);
    let outer_high2 = outer_high.saturating_mul(outer_high);
    let inner_low2 = inner_low.saturating_mul(inner_low);
    let inner_high2 = inner_high.saturating_mul(inner_high);
    let center_radius2 = center_radius.saturating_mul(center_radius);
    let outside_radius = outer_high.saturating_add(6);
    let outside_radius2 = outside_radius.saturating_mul(outside_radius);

    let mut ink_sum = 0_i64;
    let mut outer_ring_sum = 0_i64;
    let mut inner_ring_sum = 0_i64;
    let mut center_sum = 0_i64;
    let mut center_count = 0_i64;
    let mut border_ink = 0_i64;
    let mut outside_ink_sum = 0_i64;
    let mut edge_sum = 0_i64;
    let mut strong_ink_sum = 0_i64;
    let mut soft_ink_sum = 0_i64;
    let mut ring_buckets = [0_i64; QUALITY_RING_BUCKETS];
    let mut weak_wash_sum = 0_i64;

    for y in 0..image_size {
        for x in 0..image_size {
            let value_u8 = image[pixel_index(image_size, x, y)];
            let value = i64::from(value_u8);
            ink_sum = ink_sum.saturating_add(value);
            let dx = i64::try_from(x)?.saturating_sub(center);
            let dy = i64::try_from(y)?.saturating_sub(center);
            let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            if radius2 <= center_radius2 {
                center_sum = center_sum.saturating_add(value);
                center_count = center_count.saturating_add(1);
            }
            if value_u8 > STRONG_INK_THRESHOLD {
                strong_ink_sum = strong_ink_sum.saturating_add(value);
            } else if value_u8 > 0 && value_u8 <= STRONG_INK_THRESHOLD {
                soft_ink_sum = soft_ink_sum.saturating_add(value);
            }
            if value_u8 > INK_THRESHOLD {
                if x < 3 || y < 3 || x + 3 >= image_size || y + 3 >= image_size {
                    border_ink = border_ink.saturating_add(1);
                }
                if radius2 > outside_radius2 {
                    outside_ink_sum = outside_ink_sum.saturating_add(value);
                }
                if radius2 >= outer_low2 && radius2 <= outer_high2 {
                    outer_ring_sum = outer_ring_sum.saturating_add(value);
                    let bucket = angle_bucket(dx, dy);
                    ring_buckets[bucket] = ring_buckets[bucket].saturating_add(value);
                }
                if radius2 >= inner_low2 && radius2 <= inner_high2 {
                    inner_ring_sum = inner_ring_sum.saturating_add(value);
                    let bucket = angle_bucket(dx, dy);
                    ring_buckets[bucket] = ring_buckets[bucket].saturating_add(value);
                }
            }
            if value_u8 > 0 && value_u8 <= STRONG_INK_THRESHOLD {
                let neighbors = neighbor_ink_count(image, image_size, x, y);
                let near_border = x < QUALITY_EDGE_CLEAR
                    || y < QUALITY_EDGE_CLEAR
                    || x + QUALITY_EDGE_CLEAR >= image_size
                    || y + QUALITY_EDGE_CLEAR >= image_size;
                if near_border || radius2 > outside_radius2 {
                    weak_wash_sum = weak_wash_sum.saturating_add(value);
                } else if value_u8 <= INK_THRESHOLD {
                    let missing_neighbors = 4_i64.saturating_sub(i64::from(neighbors).min(4));
                    weak_wash_sum = weak_wash_sum
                        .saturating_add(value.saturating_mul(missing_neighbors.saturating_add(1)));
                } else if neighbors <= 1 {
                    weak_wash_sum = weak_wash_sum.saturating_add(value.saturating_mul(2));
                } else if neighbors <= 3 {
                    weak_wash_sum = weak_wash_sum.saturating_add(value / 2);
                }
            }
            if x + 1 < image_size {
                let right = image[pixel_index(image_size, x + 1, y)];
                let diff = abs_diff_u8(value_u8, right);
                if diff > INK_THRESHOLD {
                    edge_sum = edge_sum.saturating_add(i64::from(diff));
                }
            }
            if y + 1 < image_size {
                let down = image[pixel_index(image_size, x, y + 1)];
                let diff = abs_diff_u8(value_u8, down);
                if diff > INK_THRESHOLD {
                    edge_sum = edge_sum.saturating_add(i64::from(diff));
                }
            }
        }
    }

    let mut ring_hit_count = 0_i64;
    let mut ring_bucket_total = 0_i64;
    for &bucket_sum in &ring_buckets {
        ring_bucket_total = ring_bucket_total.saturating_add(bucket_sum);
        if bucket_sum >= 256 {
            ring_hit_count = ring_hit_count.saturating_add(1);
        }
    }
    let ring_mean = ring_bucket_total / i64::try_from(QUALITY_RING_BUCKETS)?;
    let mut ring_balance_penalty = 0_i64;
    for &bucket_sum in &ring_buckets {
        ring_balance_penalty =
            ring_balance_penalty.saturating_add(abs_i64(bucket_sum.saturating_sub(ring_mean)));
    }
    ring_balance_penalty /= 8;

    let mut speck_count = 0_i64;
    for y in 0..image_size {
        for x in 0..image_size {
            if image[pixel_index(image_size, x, y)] <= SPECK_INK_THRESHOLD {
                continue;
            }
            if neighbor_ink_count(image, image_size, x, y) <= 1 {
                speck_count = speck_count.saturating_add(1);
            }
        }
    }

    let target_ink_sum = i64::try_from(image_bytes)?.saturating_mul(47);
    let target_center_sum = center_count.saturating_mul(38);
    let density_penalty = abs_i64(ink_sum.saturating_sub(target_ink_sum)) / 3;
    let center_density_error = abs_i64(center_sum.saturating_sub(target_center_sum));
    let ring_coverage_score = ring_hit_count.saturating_mul(900);
    let ring_miss_penalty = (i64::try_from(QUALITY_RING_BUCKETS)? - ring_hit_count)
        .max(0)
        .saturating_mul(550);
    let ring_score = outer_ring_sum
        .saturating_mul(3)
        .saturating_add(inner_ring_sum.saturating_mul(2))
        .saturating_add(ring_coverage_score)
        .saturating_sub(ring_miss_penalty)
        .saturating_sub(ring_balance_penalty);
    let stroke_score = edge_sum
        .saturating_mul(2)
        .saturating_add(strong_ink_sum / 4)
        .saturating_sub(soft_ink_sum / 8);
    let interior_score = target_center_sum
        .saturating_sub(center_density_error)
        .max(0)
        .saturating_mul(6);
    let border_penalty = border_ink.saturating_mul(550);
    let outside_penalty = outside_ink_sum / 2;
    let speck_penalty = speck_count.saturating_mul(900);
    let wash_penalty = weak_wash_sum / 4;
    let total_score = ring_score
        .saturating_add(stroke_score)
        .saturating_add(interior_score)
        .saturating_sub(density_penalty)
        .saturating_sub(border_penalty)
        .saturating_sub(outside_penalty)
        .saturating_sub(speck_penalty);

    Ok(SampleQuality {
        total_score,
        ring_score,
        ring_coverage_score,
        ring_balance_penalty,
        stroke_score,
        interior_score,
        density_penalty,
        border_penalty,
        outside_penalty,
        speck_penalty,
        wash_penalty,
    })
}

fn angle_bucket(dx: i64, dy: i64) -> usize {
    let ax = abs_i64(dx);
    let ay = abs_i64(dy);
    if ax == 0 && ay == 0 {
        return 0;
    }
    let (octant, major, minor) = if dx >= 0 && dy >= 0 {
        if ax >= ay {
            (0_usize, ax, ay)
        } else {
            (1_usize, ay, ax)
        }
    } else if dx < 0 && dy >= 0 {
        if ay >= ax {
            (2_usize, ay, ax)
        } else {
            (3_usize, ax, ay)
        }
    } else if dx < 0 && dy < 0 {
        if ax >= ay {
            (4_usize, ax, ay)
        } else {
            (5_usize, ay, ax)
        }
    } else if ay >= ax {
        (6_usize, ay, ax)
    } else {
        (7_usize, ax, ay)
    };
    let local = if major == 0 {
        0
    } else {
        ((minor.saturating_mul(QUALITY_BUCKET_STEPS)) / major).min(QUALITY_BUCKET_STEPS - 1)
    };
    (octant * usize::try_from(QUALITY_BUCKET_STEPS).unwrap_or(1)
        + usize::try_from(local).unwrap_or(0))
        % QUALITY_RING_BUCKETS
}

fn neighbor_ink_count(image: &[u8], image_size: usize, x: usize, y: usize) -> u8 {
    let mut count = 0_u8;
    for dy in 0..3 {
        for dx in 0..3 {
            if dx == 1 && dy == 1 {
                continue;
            }
            let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
            let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
            if let (Some(nx), Some(ny)) = (nx, ny)
                && nx < image_size
                && ny < image_size
            {
                let value = image[pixel_index(image_size, nx, ny)];
                if value > INK_THRESHOLD {
                    count = count.saturating_add(1);
                }
            }
        }
    }
    count
}

fn clear_border(image: &mut [u8], image_size: usize, border: usize) {
    if border == 0 {
        return;
    }
    let border = border.min(image_size / 2);
    for y in 0..image_size {
        for x in 0..image_size {
            if x < border || y < border || x + border >= image_size || y + border >= image_size {
                image[pixel_index(image_size, x, y)] = 0;
            }
        }
    }
}

fn sample_signature(
    image: &[u8],
    image_size: usize,
) -> Result<[u16; SIGNATURE_BINS], Box<dyn std::error::Error>> {
    sample_signature_grid::<SIGNATURE_GRID, SIGNATURE_BINS>(image, image_size)
}

fn sample_signature_grid<const GRID: usize, const BINS: usize>(
    image: &[u8],
    image_size: usize,
) -> Result<[u16; BINS], Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!("sample has {} bytes, expected {image_bytes}", image.len()).into());
    }
    let mut sums = [0_u32; BINS];
    let mut counts = [0_u32; BINS];
    for y in 0..image_size {
        let bin_y = y * GRID / image_size;
        for x in 0..image_size {
            let bin_x = x * GRID / image_size;
            let bin = bin_y * GRID + bin_x;
            sums[bin] = sums[bin].saturating_add(u32::from(image[pixel_index(image_size, x, y)]));
            counts[bin] = counts[bin].saturating_add(1);
        }
    }
    let mut signature = [0_u16; BINS];
    for index in 0..BINS {
        if counts[index] == 0 {
            continue;
        }
        signature[index] =
            u16::try_from(sums[index].saturating_add(counts[index] / 2) / counts[index])?;
    }
    Ok(signature)
}

fn select_candidates(
    mut candidates: Vec<Candidate>,
    sample_count: usize,
    diversity_weight: i64,
) -> Result<Vec<Candidate>, Box<dyn std::error::Error>> {
    candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.source_index.cmp(&right.source_index))
    });
    if diversity_weight == 0 {
        candidates.truncate(sample_count);
        return Ok(candidates);
    }

    let mut selected = Vec::with_capacity(sample_count);
    if candidates.is_empty() {
        return Ok(selected);
    }
    selected.push(candidates.remove(0));
    while selected.len() < sample_count && !candidates.is_empty() {
        let mut best_index = 0_usize;
        let mut best_adjusted = i64::MIN;
        let mut best_score = i64::MIN;
        let mut best_source = usize::MAX;
        for (index, candidate) in candidates.iter().enumerate() {
            let distance = min_signature_distance_from_selected(&candidate.signature, &selected);
            let adjusted = candidate
                .score
                .saturating_add(distance.saturating_mul(diversity_weight));
            if adjusted > best_adjusted
                || (adjusted == best_adjusted && candidate.score > best_score)
                || (adjusted == best_adjusted
                    && candidate.score == best_score
                    && candidate.source_index < best_source)
            {
                best_index = index;
                best_adjusted = adjusted;
                best_score = candidate.score;
                best_source = candidate.source_index;
            }
        }
        selected.push(candidates.remove(best_index));
    }
    Ok(selected)
}

fn summarize_selection(
    selected: &[Candidate],
    candidate_count: usize,
    diversity_weight: i64,
) -> Result<ScoreSummary, Box<dyn std::error::Error>> {
    if selected.is_empty() {
        return Ok(ScoreSummary {
            candidate_count,
            selected_count: 0,
            diversity_weight,
            selected_min_score: 0,
            selected_max_score: 0,
            selected_mean_score_q8: 0,
            selected_min_text_distance: 0,
            selected_mean_text_distance_q8: 0,
            selected_min_signature_distance: 0,
            selected_mean_signature_distance_q8: 0,
            selected_mean_ring_score_q8: 0,
            selected_mean_ring_coverage_score_q8: 0,
            selected_mean_ring_balance_penalty_q8: 0,
            selected_mean_stroke_score_q8: 0,
            selected_mean_interior_score_q8: 0,
            selected_mean_density_penalty_q8: 0,
            selected_mean_border_penalty_q8: 0,
            selected_mean_outside_penalty_q8: 0,
            selected_mean_speck_penalty_q8: 0,
            selected_mean_wash_penalty_q8: 0,
        });
    }

    let mut score_total = 0_i64;
    let mut selected_min_score = i64::MAX;
    let mut selected_max_score = i64::MIN;
    let mut text_distance_total = 0_i64;
    let mut selected_min_text_distance = i64::MAX;
    let mut ring_score_total = 0_i64;
    let mut ring_coverage_score_total = 0_i64;
    let mut ring_balance_penalty_total = 0_i64;
    let mut stroke_score_total = 0_i64;
    let mut interior_score_total = 0_i64;
    let mut density_penalty_total = 0_i64;
    let mut border_penalty_total = 0_i64;
    let mut outside_penalty_total = 0_i64;
    let mut speck_penalty_total = 0_i64;
    let mut wash_penalty_total = 0_i64;
    for candidate in selected {
        selected_min_score = selected_min_score.min(candidate.score);
        selected_max_score = selected_max_score.max(candidate.score);
        score_total = score_total.saturating_add(candidate.score);
        selected_min_text_distance = selected_min_text_distance.min(candidate.text_distance);
        text_distance_total = text_distance_total.saturating_add(candidate.text_distance);
        ring_score_total = ring_score_total.saturating_add(candidate.quality.ring_score);
        ring_coverage_score_total =
            ring_coverage_score_total.saturating_add(candidate.quality.ring_coverage_score);
        ring_balance_penalty_total =
            ring_balance_penalty_total.saturating_add(candidate.quality.ring_balance_penalty);
        stroke_score_total = stroke_score_total.saturating_add(candidate.quality.stroke_score);
        interior_score_total =
            interior_score_total.saturating_add(candidate.quality.interior_score);
        density_penalty_total =
            density_penalty_total.saturating_add(candidate.quality.density_penalty);
        border_penalty_total =
            border_penalty_total.saturating_add(candidate.quality.border_penalty);
        outside_penalty_total =
            outside_penalty_total.saturating_add(candidate.quality.outside_penalty);
        speck_penalty_total = speck_penalty_total.saturating_add(candidate.quality.speck_penalty);
        wash_penalty_total = wash_penalty_total.saturating_add(candidate.quality.wash_penalty);
    }

    let mut min_distance = 0_i64;
    let mut mean_distance_q8 = 0_i64;
    if selected.len() > 1 {
        min_distance = i64::MAX;
        let mut distance_total = 0_i64;
        for (index, candidate) in selected.iter().enumerate() {
            let mut nearest = i64::MAX;
            for (other_index, other) in selected.iter().enumerate() {
                if index == other_index {
                    continue;
                }
                nearest = nearest.min(signature_distance(&candidate.signature, &other.signature));
            }
            min_distance = min_distance.min(nearest);
            distance_total = distance_total.saturating_add(nearest);
        }
        mean_distance_q8 = distance_total.saturating_mul(256) / i64::try_from(selected.len())?;
    }

    Ok(ScoreSummary {
        candidate_count,
        selected_count: selected.len(),
        diversity_weight,
        selected_min_score,
        selected_max_score,
        selected_mean_score_q8: score_total.saturating_mul(256) / i64::try_from(selected.len())?,
        selected_min_text_distance,
        selected_mean_text_distance_q8: mean_q8(text_distance_total, selected.len())?,
        selected_min_signature_distance: min_distance,
        selected_mean_signature_distance_q8: mean_distance_q8,
        selected_mean_ring_score_q8: mean_q8(ring_score_total, selected.len())?,
        selected_mean_ring_coverage_score_q8: mean_q8(ring_coverage_score_total, selected.len())?,
        selected_mean_ring_balance_penalty_q8: mean_q8(ring_balance_penalty_total, selected.len())?,
        selected_mean_stroke_score_q8: mean_q8(stroke_score_total, selected.len())?,
        selected_mean_interior_score_q8: mean_q8(interior_score_total, selected.len())?,
        selected_mean_density_penalty_q8: mean_q8(density_penalty_total, selected.len())?,
        selected_mean_border_penalty_q8: mean_q8(border_penalty_total, selected.len())?,
        selected_mean_outside_penalty_q8: mean_q8(outside_penalty_total, selected.len())?,
        selected_mean_speck_penalty_q8: mean_q8(speck_penalty_total, selected.len())?,
        selected_mean_wash_penalty_q8: mean_q8(wash_penalty_total, selected.len())?,
    })
}

fn mean_q8(total: i64, count: usize) -> Result<i64, Box<dyn std::error::Error>> {
    Ok(total.saturating_mul(256) / i64::try_from(count)?)
}

fn min_signature_distance_from_selected(
    signature: &[u16; SIGNATURE_BINS],
    selected: &[Candidate],
) -> i64 {
    selected
        .iter()
        .map(|candidate| signature_distance(signature, &candidate.signature))
        .min()
        .unwrap_or(0)
}

fn signature_distance(left: &[u16; SIGNATURE_BINS], right: &[u16; SIGNATURE_BINS]) -> i64 {
    let mut distance = 0_i64;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        distance = distance.saturating_add(i64::from(abs_diff_u16(*left_value, *right_value)));
    }
    distance
}

fn text_signature_distance(
    candidate: &[u16; SIGNATURE_BINS],
    target: &[u16; SIGNATURE_BINS],
) -> i64 {
    let mut best = i64::MAX;
    for variant in 0..8 {
        let mut distance = 0_i64;
        for y in 0..SIGNATURE_GRID {
            for x in 0..SIGNATURE_GRID {
                let (sx, sy) = transform_coords(SIGNATURE_GRID, x, y, variant);
                let left = candidate[pixel_index(SIGNATURE_GRID, x, y)];
                let right = target[pixel_index(SIGNATURE_GRID, sx, sy)];
                distance = distance.saturating_add(i64::from(abs_diff_u16(left, right)));
            }
        }
        best = best.min(distance);
    }
    best
}

fn apply_model_condition(
    model: &SampleModel,
    condition: usize,
    input: &[u8],
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
    feature_cache: Option<&FeatureCache>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match model {
        SampleModel::Conv3(model) => {
            let mut image = input.to_vec();
            for layer in &model.layers {
                if layer_blend_shift(&layer.condition_blend_shifts, condition) == u8::MAX {
                    continue;
                }
                image = apply_cv3_layer_condition(model, layer, condition, &image)?;
            }
            Ok(image)
        }
        SampleModel::Multichannel(model) => {
            let mut image = input.to_vec();
            for layer in &model.layers {
                if layer_blend_shift(&layer.condition_blend_shifts, condition) == u8::MAX {
                    continue;
                }
                image = apply_multichannel_layer_condition(
                    model,
                    layer,
                    condition,
                    &image,
                    target_signature,
                    feature_cache,
                )?;
            }
            Ok(image)
        }
    }
}

fn layer_blend_shift(condition_blend_shifts: &[u8], condition: usize) -> u8 {
    condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX)
}

fn apply_cv3_layer_condition(
    model: &LayeredConvModel,
    layer: &ConvLayer,
    condition: usize,
    input: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(model.image_size)?;
    if input.len() != image_bytes {
        return Err(format!(
            "input image has {} bytes, expected {image_bytes}",
            input.len()
        )
        .into());
    }
    let blend_shift = layer
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut out = vec![0_u8; input.len()];
    for y in 0..model.image_size {
        for x in 0..model.image_size {
            let index = pixel_index(model.image_size, x, y);
            let mut features = [0_i16; KERNEL];
            local_features(input, model.image_size, x, y, &mut features);
            let raw = predict_cv3_raw_pixel(
                layer,
                model.output_shift,
                condition,
                input[index],
                &features,
            );
            out[index] = blend_pixel(input[index], raw, blend_shift);
        }
    }
    Ok(out)
}

fn apply_multichannel_layer_condition(
    model: &MultichannelModel,
    layer: &MultichannelLayer,
    condition: usize,
    input: &[u8],
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
    feature_cache: Option<&FeatureCache>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(model.image_size)?;
    if input.len() != image_bytes {
        return Err(format!(
            "input image has {} bytes, expected {image_bytes}",
            input.len()
        )
        .into());
    }
    let target_signature = if model.text_conditioned {
        Some(target_signature.ok_or("NSRLTCH sampling requires a target text signature")?)
    } else {
        None
    };
    let target_stats = if feature_cache
        .and_then(|cache| cache.text_features.as_ref())
        .is_some()
    {
        None
    } else {
        target_signature.map(signature_stats)
    };
    let blend_shift = layer
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut out = vec![0_u8; input.len()];
    let mut features = vec![0_i16; model.hidden_channels];
    for y in 0..model.image_size {
        for x in 0..model.image_size {
            let index = pixel_index(model.image_size, x, y);
            conditioned_features(
                input,
                model.image_size,
                x,
                y,
                model.hidden_shift,
                target_signature,
                target_stats.as_ref(),
                feature_cache,
                &mut features,
            );
            let raw = predict_multichannel_raw_pixel(
                layer,
                model.output_shift,
                condition,
                input[index],
                &features,
            );
            out[index] = blend_pixel(input[index], raw, blend_shift);
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "arguments mirror the serialized native/WASM conditioning contract"
)]
fn conditioned_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    hidden_shift: u8,
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
    target_stats: Option<&SignatureStats>,
    feature_cache: Option<&FeatureCache>,
    out: &mut [i16],
) {
    let pixel = pixel_index(image_size, x, y);
    let image_channels = out.len().min(HIDDEN_CHANNELS);
    hidden_features(
        input,
        image_size,
        x,
        y,
        hidden_shift,
        &mut out[..image_channels],
    );
    let text_channels = if target_signature.is_some() {
        TEXT_FEATURE_CHANNELS
    } else {
        0
    };
    let new_layout = out.len() >= HIDDEN_CHANNELS + POSITION_FEATURE_CHANNELS + text_channels;
    let mut offset = HIDDEN_CHANNELS;
    if new_layout {
        if let Some(cached) = feature_cache
            .and_then(|cache| cache.position_features.get(pixel))
            .filter(|_| image_size > 0)
        {
            out[offset..offset + POSITION_FEATURE_CHANNELS].copy_from_slice(cached);
        } else {
            position_features(
                image_size,
                x,
                y,
                &mut out[offset..offset + POSITION_FEATURE_CHANNELS],
            );
        }
        offset += POSITION_FEATURE_CHANNELS;
    }
    if let Some(target_signature) = target_signature {
        if out.len() < offset + TEXT_FEATURE_CHANNELS {
            return;
        }
        let input_center = i16::from(input[pixel]);
        if let Some(text_cache) = feature_cache.and_then(|cache| cache.text_features.as_ref())
            && let (Some(static_features), Some(&center)) = (
                text_cache.static_features.get(pixel),
                text_cache.centers.get(pixel),
            )
        {
            out[offset..offset + TEXT_FEATURE_CHANNELS].copy_from_slice(static_features);
            out[offset] = center.saturating_sub(input_center).clamp(-511, 511);
            out[offset + 15] = center
                .saturating_sub(input_center)
                .saturating_add(center.saturating_sub(text_cache.global_mean))
                .saturating_mul(2)
                .clamp(-511, 511);
            return;
        }
        let Some(target_stats) = target_stats else {
            return;
        };
        text_signature_features(
            target_signature,
            target_stats,
            image_size,
            x,
            y,
            input_center,
            &mut out[offset..offset + TEXT_FEATURE_CHANNELS],
        );
    }
}

fn build_feature_cache(
    model: &SampleModel,
    image_size: usize,
    target_signature: Option<&[u16; SIGNATURE_BINS]>,
) -> Result<Option<FeatureCache>, Box<dyn std::error::Error>> {
    let SampleModel::Multichannel(model) = model else {
        return Ok(None);
    };
    let image_bytes = checked_image_bytes(image_size)?;
    let text_channels = if target_signature.is_some() {
        TEXT_FEATURE_CHANNELS
    } else {
        0
    };
    let has_position_features =
        model.hidden_channels >= HIDDEN_CHANNELS + POSITION_FEATURE_CHANNELS + text_channels;
    let mut cached_position_features = Vec::new();
    if has_position_features {
        cached_position_features.reserve(image_bytes);
        for y in 0..image_size {
            for x in 0..image_size {
                let mut features = [0_i16; POSITION_FEATURE_CHANNELS];
                position_features(image_size, x, y, &mut features);
                cached_position_features.push(features);
            }
        }
    }
    let text_features = if let Some(signature) = target_signature {
        let stats = signature_stats(signature);
        let mut centers = Vec::with_capacity(image_bytes);
        let mut static_features = Vec::with_capacity(image_bytes);
        for y in 0..image_size {
            for x in 0..image_size {
                let center = interpolated_signature_value(signature, image_size, x, y);
                let mut features = [0_i16; TEXT_FEATURE_CHANNELS];
                text_signature_features(signature, &stats, image_size, x, y, 0, &mut features);
                centers.push(center);
                static_features.push(features);
            }
        }
        Some(TextFeatureCache {
            centers,
            global_mean: stats.global_mean,
            static_features,
        })
    } else {
        None
    };
    if cached_position_features.is_empty() && text_features.is_none() {
        Ok(None)
    } else {
        Ok(Some(FeatureCache {
            position_features: cached_position_features,
            text_features,
        }))
    }
}

fn interpolated_signature_value(
    signature: &[u16; SIGNATURE_BINS],
    image_size: usize,
    x: usize,
    y: usize,
) -> i16 {
    if image_size <= 1 {
        return i16::try_from(signature[0].min(255)).unwrap_or(0);
    }
    let grid_max = SIGNATURE_GRID - 1;
    let scale = 256_usize;
    let sx = x.saturating_mul(grid_max).saturating_mul(scale) / (image_size - 1);
    let sy = y.saturating_mul(grid_max).saturating_mul(scale) / (image_size - 1);
    let x0 = (sx / scale).min(grid_max);
    let y0 = (sy / scale).min(grid_max);
    let x1 = (x0 + 1).min(grid_max);
    let y1 = (y0 + 1).min(grid_max);
    let wx = i64::try_from(sx % scale).unwrap_or(0);
    let wy = i64::try_from(sy % scale).unwrap_or(0);
    let ix = i64::try_from(scale).unwrap_or(256).saturating_sub(wx);
    let iy = i64::try_from(scale).unwrap_or(256).saturating_sub(wy);
    let at =
        |xx: usize, yy: usize| -> i64 { i64::from(signature[yy * SIGNATURE_GRID + xx].min(255)) };
    let weighted = at(x0, y0)
        .saturating_mul(ix)
        .saturating_mul(iy)
        .saturating_add(at(x1, y0).saturating_mul(wx).saturating_mul(iy))
        .saturating_add(at(x0, y1).saturating_mul(ix).saturating_mul(wy))
        .saturating_add(at(x1, y1).saturating_mul(wx).saturating_mul(wy));
    i16::try_from((weighted + 32_768) >> 16).unwrap_or(0)
}

fn text_signature_features(
    signature: &[u16; SIGNATURE_BINS],
    stats: &SignatureStats,
    image_size: usize,
    x: usize,
    y: usize,
    input_center: i16,
    out: &mut [i16],
) {
    if out.len() < TEXT_FEATURE_CHANNELS {
        return;
    }
    let step = (image_size / SIGNATURE_GRID.max(1)).max(1);
    let center = interpolated_signature_value(signature, image_size, x, y);
    let left = interpolated_signature_value(signature, image_size, x.saturating_sub(step), y);
    let right = interpolated_signature_value(
        signature,
        image_size,
        (x + step).min(image_size.saturating_sub(1)),
        y,
    );
    let up = interpolated_signature_value(signature, image_size, x, y.saturating_sub(step));
    let down = interpolated_signature_value(
        signature,
        image_size,
        x,
        (y + step).min(image_size.saturating_sub(1)),
    );
    let up_left = interpolated_signature_value(
        signature,
        image_size,
        x.saturating_sub(step),
        y.saturating_sub(step),
    );
    let down_right = interpolated_signature_value(
        signature,
        image_size,
        (x + step).min(image_size.saturating_sub(1)),
        (y + step).min(image_size.saturating_sub(1)),
    );
    let neighbor_mean =
        i16::try_from((i32::from(left) + i32::from(right) + i32::from(up) + i32::from(down)) / 4)
            .unwrap_or(0);
    let grid_x = signature_grid_coord(image_size, x);
    let grid_y = signature_grid_coord(image_size, y);
    let global_mean = stats.global_mean;
    let row_mean = stats.row_means[grid_y];
    let col_mean = stats.col_means[grid_x];
    let horizontal_curve = left
        .saturating_add(right)
        .saturating_sub(center.saturating_mul(2));
    let vertical_curve = up
        .saturating_add(down)
        .saturating_sub(center.saturating_mul(2));
    out[0] = center.saturating_sub(input_center).clamp(-511, 511);
    out[1] = center.saturating_sub(64).clamp(-511, 511);
    out[2] = center.saturating_sub(32).clamp(-511, 511);
    out[3] = center.saturating_sub(48).saturating_mul(4).clamp(-511, 511);
    out[4] = right
        .saturating_sub(left)
        .saturating_mul(2)
        .clamp(-511, 511);
    out[5] = down.saturating_sub(up).saturating_mul(2).clamp(-511, 511);
    out[6] = down_right
        .saturating_sub(up_left)
        .saturating_mul(2)
        .clamp(-511, 511);
    out[7] = center
        .saturating_sub(neighbor_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[8] = center
        .saturating_sub(global_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[9] = center
        .saturating_sub(row_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[10] = center
        .saturating_sub(col_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[11] = horizontal_curve.saturating_mul(4).clamp(-511, 511);
    out[12] = vertical_curve.saturating_mul(4).clamp(-511, 511);
    out[13] = if center >= 160 { 256 } else { -256 };
    out[14] = if center <= 32 { -256 } else { 256 };
    out[15] = center
        .saturating_sub(input_center)
        .saturating_add(center.saturating_sub(global_mean))
        .saturating_mul(2)
        .clamp(-511, 511);
}

fn signature_grid_coord(image_size: usize, value: usize) -> usize {
    if image_size <= 1 {
        return 0;
    }
    value
        .saturating_mul(SIGNATURE_GRID)
        .checked_div(image_size)
        .unwrap_or(0)
        .min(SIGNATURE_GRID - 1)
}

fn signature_stats(signature: &[u16; SIGNATURE_BINS]) -> SignatureStats {
    let global_mean = signature_global_mean(signature);
    let mut row_means = [0_i16; SIGNATURE_GRID];
    let mut col_means = [0_i16; SIGNATURE_GRID];
    for (row, mean) in row_means.iter_mut().enumerate() {
        *mean = signature_row_mean(signature, row);
    }
    for (col, mean) in col_means.iter_mut().enumerate() {
        *mean = signature_col_mean(signature, col);
    }
    SignatureStats {
        global_mean,
        row_means,
        col_means,
    }
}

fn signature_global_mean(signature: &[u16; SIGNATURE_BINS]) -> i16 {
    let total: u32 = signature
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(SIGNATURE_BINS).unwrap_or(1)).unwrap_or(0)
}

fn signature_row_mean(signature: &[u16; SIGNATURE_BINS], row: usize) -> i16 {
    let row = row.min(SIGNATURE_GRID - 1);
    let start = row * SIGNATURE_GRID;
    let total: u32 = signature[start..start + SIGNATURE_GRID]
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
}

fn signature_col_mean(signature: &[u16; SIGNATURE_BINS], col: usize) -> i16 {
    let col = col.min(SIGNATURE_GRID - 1);
    let mut total = 0_u32;
    for row in 0..SIGNATURE_GRID {
        total = total.saturating_add(u32::from(signature[row * SIGNATURE_GRID + col].min(255)));
    }
    i16::try_from(total / u32::try_from(SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
}

fn position_features(image_size: usize, x: usize, y: usize, out: &mut [i16]) {
    if out.len() < POSITION_FEATURE_CHANNELS || image_size == 0 {
        return;
    }
    let size = i64::try_from(image_size).unwrap_or(1).max(1);
    let dx = i64::try_from(x)
        .unwrap_or(0)
        .saturating_mul(2)
        .saturating_add(1)
        .saturating_sub(size);
    let dy = i64::try_from(y)
        .unwrap_or(0)
        .saturating_mul(2)
        .saturating_add(1)
        .saturating_sub(size);
    let radius = integer_sqrt_u64(
        u64::try_from(dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy))).unwrap_or(0),
    );
    let radius_i64 = i64::try_from(radius).unwrap_or(i64::MAX);
    let outer_radius = size.saturating_sub(18).max(1);
    let inner_radius = size.saturating_sub(38).max(1);
    let ring_width = (size / 12).max(6);

    out[0] = clamp_i16(dx.saturating_mul(256) / size);
    out[1] = clamp_i16(dy.saturating_mul(256) / size);
    out[2] = clamp_i16(radius_i64.saturating_mul(384) / size - 192);
    out[3] = clamp_i16(triangular_peak(radius_i64, outer_radius, ring_width));
    out[4] = clamp_i16(triangular_peak(radius_i64, inner_radius, ring_width));
    out[5] = if radius_i64 <= outer_radius {
        192
    } else {
        -192
    };
}

fn triangular_peak(value: i64, center: i64, width: i64) -> i64 {
    let distance = abs_i64(value.saturating_sub(center));
    if distance >= width {
        return 0;
    }
    255 - distance.saturating_mul(255) / width.max(1)
}

fn local_features(input: &[u8], image_size: usize, x: usize, y: usize, out: &mut [i16; KERNEL]) {
    let center_index = pixel_index(image_size, x, y);
    let center = i16::from(input[center_index]);
    if x > 0 && y > 0 && x + 1 < image_size && y + 1 < image_size {
        let above = center_index - image_size;
        let below = center_index + image_size;
        out[0] = i16::from(input[above - 1]) - center;
        out[1] = i16::from(input[above]) - center;
        out[2] = i16::from(input[above + 1]) - center;
        out[3] = i16::from(input[center_index - 1]) - center;
        out[4] = 0;
        out[5] = i16::from(input[center_index + 1]) - center;
        out[6] = i16::from(input[below - 1]) - center;
        out[7] = i16::from(input[below]) - center;
        out[8] = i16::from(input[below + 1]) - center;
        return;
    }
    let mut index = 0_usize;
    for dy in 0..3 {
        for dx in 0..3 {
            let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
            let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
            let neighbor = if let (Some(nx), Some(ny)) = (nx, ny) {
                if nx < image_size && ny < image_size {
                    i16::from(input[pixel_index(image_size, nx, ny)])
                } else {
                    center
                }
            } else {
                center
            };
            out[index] = neighbor - center;
            index += 1;
        }
    }
}

fn predict_cv3_raw_pixel(
    layer: &ConvLayer,
    output_shift: u8,
    condition: usize,
    input_center: u8,
    features: &[i16; KERNEL],
) -> u8 {
    let mut acc = i64::from(*layer.condition_biases.get(condition).unwrap_or(&0));
    for (weight, feature) in layer.weights.iter().zip(features.iter()) {
        acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
    }
    let residual = signed_round_shift(acc, output_shift);
    let predicted = i64::from(input_center).saturating_add(residual);
    predicted.clamp(0, i64::from(u8::MAX)) as u8
}

fn hidden_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    hidden_shift: u8,
    out: &mut [i16],
) {
    let mut local = [0_i16; KERNEL];
    local_features(input, image_size, x, y, &mut local);
    for (channel, kernel) in HIDDEN_KERNELS.iter().take(out.len()).enumerate() {
        let mut acc = 0_i64;
        for (weight, feature) in kernel.iter().zip(local.iter()) {
            acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
        }
        out[channel] = signed_round_shift(acc, hidden_shift).clamp(-511, 511) as i16;
    }
}

fn predict_multichannel_raw_pixel(
    layer: &MultichannelLayer,
    output_shift: u8,
    condition: usize,
    input_center: u8,
    features: &[i16],
) -> u8 {
    let mut acc = i64::from(*layer.condition_biases.get(condition).unwrap_or(&0));
    if let Some(weights) = layer.condition_weights.get(condition) {
        for (weight, feature) in weights.iter().zip(features.iter()) {
            acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
        }
    }
    let residual = signed_round_shift(acc, output_shift);
    let predicted = i64::from(input_center).saturating_add(residual);
    predicted.clamp(0, i64::from(u8::MAX)) as u8
}

fn blend_pixel(input: u8, raw: u8, blend_shift: u8) -> u8 {
    if blend_shift == u8::MAX {
        return input;
    }
    if blend_shift == 0 {
        return raw;
    }
    let delta = i16::from(raw) - i16::from(input);
    let rounding = 1_i16 << (blend_shift - 1);
    let adjustment = if delta >= 0 {
        (delta + rounding) >> blend_shift
    } else {
        -(((-delta) + rounding) >> blend_shift)
    };
    let out = i16::from(input) + adjustment;
    out.clamp(0, i16::from(u8::MAX)) as u8
}

fn signed_round_shift(value: i64, shift: u8) -> i64 {
    if shift == 0 {
        return value;
    }
    let rounding = 1_i64 << (shift - 1);
    if value >= 0 {
        value.saturating_add(rounding) >> shift
    } else {
        -((value.saturating_neg().saturating_add(rounding)) >> shift)
    }
}

fn integer_sqrt_u64(value: u64) -> u64 {
    if value <= 1 {
        return value;
    }
    let mut low = 1_u64;
    let mut high = value.min(1_u64 << 32);
    while low <= high {
        let mid = low + ((high - low) / 2);
        let square = u128::from(mid) * u128::from(mid);
        if square == u128::from(value) {
            return mid;
        }
        if square < u128::from(value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    high
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(-511, 511) as i16
}

fn abs_i64(value: i64) -> i64 {
    if value >= 0 {
        value
    } else {
        value.saturating_neg()
    }
}

fn abs_diff_u8(left: u8, right: u8) -> u8 {
    left.abs_diff(right)
}

fn abs_diff_u16(left: u16, right: u16) -> u16 {
    left.abs_diff(right)
}

fn pixel_index(image_size: usize, x: usize, y: usize) -> usize {
    y * image_size + x
}

fn checked_image_bytes(image_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image byte count overflow".into())
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.state = value;
        value
    }

    fn rand_bounded(&mut self, max_exclusive: u32) -> u32 {
        if max_exclusive <= 1 {
            return 0;
        }
        let range = 1_u64 << 32;
        let bound = u64::from(max_exclusive);
        let limit = range - (range % bound);
        loop {
            let value = u64::from(self.next_u32());
            if value < limit {
                return (value % bound) as u32;
            }
        }
    }
}

fn hash_seed(parts: &[&str]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for part in parts {
        for byte in part.bytes() {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(16_777_619);
        }
        hash ^= 255;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash | 1
}

struct ContactSheet {
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}

fn make_contact_sheet(
    images: &[Vec<u8>],
    tile_size: usize,
    columns: usize,
    gap: usize,
) -> ContactSheet {
    let rows = images.len().div_ceil(columns);
    let width = columns * tile_size + (columns + 1) * gap;
    let height = rows * tile_size + (rows + 1) * gap;
    let mut sheet = vec![255_u8; width * height];
    for (image_index, image) in images.iter().enumerate() {
        let col = image_index % columns;
        let row = image_index / columns;
        let x0 = gap + col * (tile_size + gap);
        let y0 = gap + row * (tile_size + gap);
        for y in 0..tile_size {
            for x in 0..tile_size {
                let value = image[pixel_index(tile_size, x, y)];
                sheet[pixel_index(width, x0 + x, y0 + y)] = 255_u8.saturating_sub(value);
            }
        }
    }
    ContactSheet {
        width,
        height,
        bytes: sheet,
    }
}

fn write_pgm(
    path: &Path,
    width: usize,
    height: usize,
    bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{width} {height}\n255\n")?;
    file.write_all(bytes)?;
    Ok(())
}

fn write_trace(
    config: &Config,
    model: &SampleModel,
    conditions: &[usize],
    raw_path: &Path,
    pgm_path: &Path,
    latent_condition: Option<&LatentCondition>,
    score_summary: &ScoreSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str("{\n");
    json_field(&mut out, "schema", SCHEMA, true);
    json_field(
        &mut out,
        "model",
        &config.model_path.display().to_string(),
        true,
    );
    json_field(&mut out, "model_format", model.format_name(), true);
    json_field(&mut out, "init_mode", config.init_mode.as_str(), true);
    json_field(&mut out, "seed", &config.seed, true);
    number_field(&mut out, "image_size", model.image_size(), true);
    number_field(&mut out, "timesteps", model.timesteps(), true);
    number_field(&mut out, "layers", model.layer_count(), true);
    number_field(&mut out, "feature_channels", model.feature_channels(), true);
    number_field(&mut out, "samples", score_summary.selected_count, true);
    number_field(
        &mut out,
        "candidate_multiplier",
        config.candidate_multiplier,
        true,
    );
    number_field(
        &mut out,
        "candidate_count",
        score_summary.candidate_count,
        true,
    );
    number_field(&mut out, "worker_count", config.worker_count, true);
    number_field(
        &mut out,
        "selected_count",
        score_summary.selected_count,
        true,
    );
    signed_number_field(
        &mut out,
        "diversity_weight",
        score_summary.diversity_weight,
        true,
    );
    number_field(&mut out, "border_clear", config.border_clear, true);
    signed_number_field(
        &mut out,
        "selected_min_score",
        score_summary.selected_min_score,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_max_score",
        score_summary.selected_max_score,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_score_q8",
        score_summary.selected_mean_score_q8,
        true,
    );
    let latent_model = latent_condition
        .map(|condition| condition.model_path.display().to_string())
        .unwrap_or_default();
    let latent_target_plan = latent_condition
        .map(|condition| condition.target_plan_path.display().to_string())
        .unwrap_or_default();
    let latent_prompt = latent_condition
        .map(|condition| condition.prompt.clone())
        .unwrap_or_default();
    json_field(&mut out, "latent_model", &latent_model, true);
    json_field(&mut out, "latent_target_plan", &latent_target_plan, true);
    json_field(&mut out, "latent_prompt", &latent_prompt, true);
    let latent_target_source = latent_condition
        .map(|condition| condition.target_source.clone())
        .unwrap_or_default();
    let latent_target_name = latent_condition
        .map(|condition| condition.target_name.clone())
        .unwrap_or_default();
    json_field(
        &mut out,
        "latent_target_source",
        &latent_target_source,
        true,
    );
    number_field(
        &mut out,
        "latent_target_number",
        latent_condition
            .map(|condition| condition.target_number)
            .unwrap_or(0),
        true,
    );
    json_field(&mut out, "latent_target_name", &latent_target_name, true);
    signed_number_field(
        &mut out,
        "latent_target_score",
        latent_condition
            .map(|condition| condition.target_score)
            .unwrap_or(0),
        true,
    );
    signed_number_field(
        &mut out,
        "latent_target_latent_score",
        latent_condition
            .map(|condition| condition.target_latent_score)
            .unwrap_or(0),
        true,
    );
    signed_number_field(
        &mut out,
        "latent_target_lexical_score",
        latent_condition
            .map(|condition| condition.target_lexical_score)
            .unwrap_or(0),
        true,
    );
    number_field(
        &mut out,
        "latent_dim",
        latent_condition
            .map(|condition| condition.latent_dim)
            .unwrap_or(0),
        true,
    );
    number_field(
        &mut out,
        "latent_text_features",
        latent_condition
            .map(|condition| condition.text_feature_count)
            .unwrap_or(0),
        true,
    );
    match latent_condition {
        Some(condition) => number_array_field(
            &mut out,
            "latent_target_signature",
            &condition.target_signature,
            true,
        ),
        None => empty_array_field(&mut out, "latent_target_signature", true),
    }
    signed_number_field(&mut out, "text_weight", config.text_weight, true);
    signed_number_field(
        &mut out,
        "selected_min_text_distance",
        score_summary.selected_min_text_distance,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_text_distance_q8",
        score_summary.selected_mean_text_distance_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_min_signature_distance",
        score_summary.selected_min_signature_distance,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_signature_distance_q8",
        score_summary.selected_mean_signature_distance_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_ring_score_q8",
        score_summary.selected_mean_ring_score_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_ring_coverage_score_q8",
        score_summary.selected_mean_ring_coverage_score_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_ring_balance_penalty_q8",
        score_summary.selected_mean_ring_balance_penalty_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_stroke_score_q8",
        score_summary.selected_mean_stroke_score_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_interior_score_q8",
        score_summary.selected_mean_interior_score_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_density_penalty_q8",
        score_summary.selected_mean_density_penalty_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_border_penalty_q8",
        score_summary.selected_mean_border_penalty_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_outside_penalty_q8",
        score_summary.selected_mean_outside_penalty_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_speck_penalty_q8",
        score_summary.selected_mean_speck_penalty_q8,
        true,
    );
    signed_number_field(
        &mut out,
        "selected_mean_wash_penalty_q8",
        score_summary.selected_mean_wash_penalty_q8,
        true,
    );
    number_field(&mut out, "passes", config.passes, true);
    json_field(
        &mut out,
        "raw_samples",
        &raw_path.display().to_string(),
        true,
    );
    json_field(
        &mut out,
        "preview_pgm",
        &pgm_path.display().to_string(),
        true,
    );
    out.push_str("  \"active_conditions\":[");
    for (index, condition) in conditions.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        let kind_index = condition / model.timesteps();
        let timestep = condition % model.timesteps() + 1;
        let kind = CORRUPTION_KINDS
            .get(kind_index)
            .copied()
            .unwrap_or("unknown");
        out.push_str(&format!(
            "{{\"condition\":{},\"corruption\":\"{}\",\"timestep\":{}}}",
            condition,
            json_escape(kind),
            timestep
        ));
    }
    out.push_str("]\n");
    out.push_str("}\n");
    fs::write(config.out_dir.join("trace.json"), out)?;
    Ok(())
}

fn json_field(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&json_escape(value));
    out.push('"');
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn number_field(out: &mut String, key: &str, value: usize, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn signed_number_field(out: &mut String, key: &str, value: i64, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn number_array_field(out: &mut String, key: &str, values: &[u16], comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn empty_array_field(out: &mut String, key: &str, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":[]");
    if comma {
        out.push(',');
    }
    out.push('\n');
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
            _ => out.push(ch),
        }
    }
    out
}
