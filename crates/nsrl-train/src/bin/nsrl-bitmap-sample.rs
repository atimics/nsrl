#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "nsrl.bitmap_sampler_trace.v1";
const MODEL_MAGIC_CV3: &[u8; 8] = b"NSRLCV3\n";
const MODEL_MAGIC_MCH: &[u8; 8] = b"NSRLMCH\n";
const KERNEL: usize = 9;
const HIDDEN_CHANNELS: usize = 8;
const SIGNATURE_GRID: usize = 8;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
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
const CORRUPTION_KINDS: [&str; 8] = [
    "pixel-dropout",
    "salt-pepper",
    "block-mask",
    "stroke-thin",
    "stroke-thicken",
    "line-drop",
    "mixed-noise",
    "coarse-erase",
];

#[derive(Debug, Clone)]
struct Config {
    model_path: PathBuf,
    out_dir: PathBuf,
    prior_clean_path: PathBuf,
    sample_count: usize,
    candidate_multiplier: usize,
    diversity_weight: i64,
    border_clear: usize,
    passes: usize,
    preview_columns: usize,
    seed: String,
    init_mode: InitMode,
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
            prior_clean_path: PathBuf::from(
                "data/processed/key-solomon-goetia-denoise-v1/clean/train.ink128.u8",
            ),
            sample_count: 64,
            candidate_multiplier: 1,
            diversity_weight: 0,
            border_clear: 0,
            passes: 8,
            preview_columns: 8,
            seed: "solomon-sampler-v1".to_string(),
            init_mode: InitMode::SealPrior,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitMode {
    Noise,
    SealPrior,
    LearnedPrior,
    PatchPrior,
    CoordinatePrior,
}

impl InitMode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "noise" => Ok(Self::Noise),
            "seal-prior" => Ok(Self::SealPrior),
            "learned-prior" => Ok(Self::LearnedPrior),
            "patch-prior" => Ok(Self::PatchPrior),
            "coordinate-prior" => Ok(Self::CoordinatePrior),
            _ => Err(format!(
                "unknown init mode: {value}; expected noise, seal-prior, learned-prior, patch-prior, or coordinate-prior"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Noise => "noise",
            Self::SealPrior => "seal-prior",
            Self::LearnedPrior => "learned-prior",
            Self::PatchPrior => "patch-prior",
            Self::CoordinatePrior => "coordinate-prior",
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
            Self::Multichannel(_) => "NSRLMCH",
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

#[derive(Debug)]
struct CleanPrior {
    path: PathBuf,
    images: Vec<u8>,
    coordinate_sums: Vec<u32>,
    image_count: usize,
}

#[derive(Debug)]
struct Candidate {
    source_index: usize,
    score: i64,
    signature: [u16; SIGNATURE_BINS],
    image: Vec<u8>,
}

#[derive(Debug)]
struct ScoreSummary {
    candidate_count: usize,
    selected_count: usize,
    diversity_weight: i64,
    selected_min_score: i64,
    selected_max_score: i64,
    selected_mean_score_q8: i64,
    selected_min_signature_distance: i64,
    selected_mean_signature_distance_q8: i64,
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
    fs::create_dir_all(&config.out_dir)?;

    let model = read_model(&config.model_path)?;
    let conditions = active_conditions(&model);
    if conditions.is_empty() {
        return Err("model has no active blend conditions".into());
    }

    let image_size = model.image_size();
    let prior = if config.init_mode == InitMode::LearnedPrior
        || config.init_mode == InitMode::PatchPrior
        || config.init_mode == InitMode::CoordinatePrior
    {
        Some(read_clean_prior(&config.prior_clean_path, image_size)?)
    } else {
        None
    };
    let image_bytes = checked_image_bytes(image_size)?;
    let candidate_count = config
        .sample_count
        .checked_mul(config.candidate_multiplier)
        .ok_or("candidate count overflow")?;
    let mut candidates = Vec::with_capacity(candidate_count);
    for candidate_index in 0..candidate_count {
        let mut image = initial_image(&config, image_size, candidate_index, prior.as_ref())?;
        for _ in 0..config.passes {
            for &condition in &conditions {
                image = apply_model_condition(&model, condition, &image)?;
            }
        }
        clear_border(&mut image, image_size, config.border_clear);
        let score = score_sample(&image, image_size)?;
        let signature = sample_signature(&image, image_size)?;
        candidates.push(Candidate {
            source_index: candidate_index,
            score,
            signature,
            image,
        });
    }
    let selected = select_candidates(candidates, config.sample_count, config.diversity_weight)?;
    let score_summary = summarize_selection(&selected, candidate_count, config.diversity_weight)?;

    let mut samples = Vec::with_capacity(config.sample_count * image_bytes);
    let mut preview = Vec::new();
    for candidate in selected {
        samples.extend_from_slice(&candidate.image);
        preview.push(candidate.image);
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
        prior.as_ref(),
        &score_summary,
    )?;

    println!(
        "{{\"schema\":\"{}\",\"samples\":{},\"passes\":{},\"init_mode\":\"{}\",\"out_dir\":\"{}\",\"pgm\":\"{}\"}}",
        SCHEMA,
        config.sample_count,
        config.passes,
        config.init_mode.as_str(),
        json_escape(&config.out_dir.display().to_string()),
        json_escape(&pgm_path.display().to_string()),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-bitmap-sample [--model PATH] [--out-dir PATH] [--prior-clean PATH] [--samples N] [--candidate-multiplier N] [--diversity-weight N] [--border-clear N] [--passes N] [--preview-columns N] [--seed TEXT] [--init noise|seal-prior|learned-prior|patch-prior|coordinate-prior]"
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
            "--prior-clean" => {
                config.prior_clean_path =
                    PathBuf::from(args.next().ok_or("--prior-clean requires PATH")?);
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
            "--seed" => {
                config.seed = args.next().ok_or("--seed requires TEXT")?;
            }
            "--init" => {
                config.init_mode = InitMode::parse(&args.next().ok_or("--init requires MODE")?)?;
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
        SampleModel::Multichannel(read_multichannel_model(&mut cursor)?)
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
) -> Result<MultichannelModel, Box<dyn std::error::Error>> {
    let image_size = usize::try_from(cursor.read_u32()?)?;
    let timesteps = usize::try_from(cursor.read_u32()?)?;
    let hidden_shift = u8::try_from(cursor.read_u32()?)?;
    let output_shift = u8::try_from(cursor.read_u32()?)?;
    let hidden_channels = usize::try_from(cursor.read_u32()?)?;
    if hidden_channels == 0 || hidden_channels > HIDDEN_KERNELS.len() {
        return Err(format!(
            "unsupported hidden channel count: {hidden_channels}; sampler supports {}",
            HIDDEN_KERNELS.len()
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

fn read_clean_prior(
    path: &Path,
    image_size: usize,
) -> Result<CleanPrior, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let images = fs::read(path)?;
    if images.is_empty() {
        return Err(format!("{} is empty", path.display()).into());
    }
    if images.len() % image_bytes != 0 {
        return Err(format!(
            "{} has {} bytes, which is not a multiple of {image_bytes}",
            path.display(),
            images.len()
        )
        .into());
    }
    let image_count = images.len() / image_bytes;
    let mut coordinate_sums = vec![0_u32; image_bytes];
    for image in images.chunks_exact(image_bytes) {
        for (sum, value) in coordinate_sums.iter_mut().zip(image.iter()) {
            *sum = sum.saturating_add(u32::from(*value));
        }
    }
    Ok(CleanPrior {
        path: path.to_path_buf(),
        images,
        coordinate_sums,
        image_count,
    })
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

fn initial_image(
    config: &Config,
    image_size: usize,
    sample_index: usize,
    prior: Option<&CleanPrior>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let seed = hash_seed(&[
        &config.seed,
        &sample_index.to_string(),
        config.init_mode.as_str(),
    ]);
    let mut rng = Rng::new(seed);
    match config.init_mode {
        InitMode::Noise => Ok(initial_noise(image_size, &mut rng)?),
        InitMode::SealPrior => Ok(initial_seal_prior(image_size, &mut rng)?),
        InitMode::LearnedPrior => {
            let prior = prior.ok_or("learned-prior init requires --prior-clean")?;
            Ok(initial_learned_prior(image_size, &mut rng, prior)?)
        }
        InitMode::PatchPrior => {
            let prior = prior.ok_or("patch-prior init requires --prior-clean")?;
            Ok(initial_patch_prior(image_size, &mut rng, prior)?)
        }
        InitMode::CoordinatePrior => {
            let prior = prior.ok_or("coordinate-prior init requires --prior-clean")?;
            Ok(initial_coordinate_prior(image_size, &mut rng, prior)?)
        }
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

fn initial_seal_prior(
    image_size: usize,
    rng: &mut Rng,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let mut image = vec![0_u8; image_bytes];
    let center = i32::try_from(image_size / 2)?;
    let jitter_x = i32::try_from(rng.rand_bounded(9))? - 4;
    let jitter_y = i32::try_from(rng.rand_bounded(9))? - 4;
    let cx = center + jitter_x;
    let cy = center + jitter_y;
    let base_radius = i32::try_from(image_size / 2)? - 13;
    let outer = base_radius + i32::try_from(rng.rand_bounded(7))? - 3;
    draw_circle(&mut image, image_size, cx, cy, outer, 2, 210)?;
    draw_circle(
        &mut image,
        image_size,
        cx,
        cy,
        outer - 8 - i32::try_from(rng.rand_bounded(4))?,
        1,
        190,
    )?;

    let stroke_count = 5 + usize::try_from(rng.rand_bounded(5))?;
    let span = outer - 8;
    for _ in 0..stroke_count {
        let x0 = cx + random_signed(rng, span)?;
        let y0 = cy + random_signed(rng, span)?;
        let x1 = cx + random_signed(rng, span)?;
        let y1 = cy + random_signed(rng, span)?;
        let thickness = 1 + i32::try_from(rng.rand_bounded(2))?;
        draw_line(&mut image, image_size, x0, y0, x1, y1, thickness, 230)?;
    }

    for _ in 0..3 {
        let arm = 12 + i32::try_from(rng.rand_bounded(18))?;
        let x = cx + random_signed(rng, span / 2)?;
        let y = cy + random_signed(rng, span / 2)?;
        draw_line(&mut image, image_size, x - arm, y, x + arm, y, 1, 240)?;
        draw_line(&mut image, image_size, x, y - arm, x, y + arm, 1, 240)?;
    }

    for value in &mut image {
        if *value > 0 && rng.chance(1, 18) {
            *value = value.saturating_sub(80);
        } else if *value == 0 && rng.chance(1, 96) {
            *value = 160;
        }
    }
    Ok(image)
}

fn initial_learned_prior(
    image_size: usize,
    rng: &mut Rng,
    prior: &CleanPrior,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let donor_index = usize::try_from(rng.rand_bounded(u32::try_from(prior.image_count)?))?;
    let donor = clean_prior_image(prior, image_bytes, donor_index)?;
    let variant = usize::try_from(rng.rand_bounded(8))?;
    let mut image = transformed_prior_image(donor, image_size, variant)?;
    let shift_x = random_signed(rng, 6)?;
    let shift_y = random_signed(rng, 6)?;
    image = shift_bitmap(&image, image_size, shift_x, shift_y)?;

    if prior.image_count > 1 && rng.chance(1, 3) {
        let other_index = usize::try_from(rng.rand_bounded(u32::try_from(prior.image_count)?))?;
        let other = clean_prior_image(prior, image_bytes, other_index)?;
        let other_variant = usize::try_from(rng.rand_bounded(8))?;
        let mut other_image = transformed_prior_image(other, image_size, other_variant)?;
        let other_shift_x = random_signed(rng, 9)?;
        let other_shift_y = random_signed(rng, 9)?;
        other_image = shift_bitmap(&other_image, image_size, other_shift_x, other_shift_y)?;
        overlay_prior(&mut image, &other_image, rng);
    }

    if prior.image_count > 1 {
        splice_prior_rectangles(&mut image, image_size, rng, prior)?;
    }
    if rng.chance(1, 2) {
        image = dilate_bitmap(&image, image_size);
    }
    jitter_prior(&mut image, rng);
    Ok(image)
}

fn initial_coordinate_prior(
    image_size: usize,
    rng: &mut Rng,
    prior: &CleanPrior,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let count = u32::try_from(prior.image_count)?;
    let half_count = count / 2;
    let mut image = vec![0_u8; image_bytes];
    for (index, value) in image.iter_mut().enumerate() {
        let mean_ink = prior.coordinate_sums[index].saturating_add(half_count) / count;
        if rng.rand_bounded(1024) < mean_ink {
            *value = u8::try_from(144 + rng.rand_bounded(112))?;
        } else if rng.chance(1, 512) {
            *value = 96;
        }
    }
    if rng.chance(1, 4) {
        image = dilate_bitmap(&image, image_size);
    }
    jitter_coordinate_prior(&mut image, rng);
    Ok(image)
}

fn initial_patch_prior(
    image_size: usize,
    rng: &mut Rng,
    prior: &CleanPrior,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let tile_size = (image_size / 4).max(16);
    let global_variant = usize::try_from(rng.rand_bounded(8))?;
    let mut image = vec![0_u8; image_bytes];
    let mut weight = vec![0_u16; image_bytes];
    let mut tile_y = 0_usize;
    while tile_y < image_size {
        let tile_h = (image_size - tile_y).min(tile_size);
        let mut tile_x = 0_usize;
        while tile_x < image_size {
            let tile_w = (image_size - tile_x).min(tile_size);
            let donor_index = usize::try_from(rng.rand_bounded(u32::try_from(prior.image_count)?))?;
            let donor = clean_prior_image(prior, image_bytes, donor_index)?;
            add_patch_tile(
                &mut image,
                &mut weight,
                donor,
                image_size,
                global_variant,
                tile_x,
                tile_y,
                tile_w,
                tile_h,
            );
            tile_x += tile_size;
        }
        tile_y += tile_size;
    }

    let overlay_count = 2 + usize::try_from(rng.rand_bounded(3))?;
    for _ in 0..overlay_count {
        let donor_index = usize::try_from(rng.rand_bounded(u32::try_from(prior.image_count)?))?;
        let donor = clean_prior_image(prior, image_bytes, donor_index)?;
        let variant = usize::try_from(rng.rand_bounded(8))?;
        let patch = 16 + usize::try_from(rng.rand_bounded(25))?;
        let max_start = image_size.saturating_sub(patch);
        let x0 = usize::try_from(rng.rand_bounded(u32::try_from(max_start + 1)?))?;
        let y0 = usize::try_from(rng.rand_bounded(u32::try_from(max_start + 1)?))?;
        stamp_patch(&mut image, donor, image_size, variant, x0, y0, patch, rng);
    }

    if rng.chance(1, 2) {
        image = dilate_bitmap(&image, image_size);
    }
    jitter_patch_prior(&mut image, rng);
    Ok(image)
}

fn add_patch_tile(
    image: &mut [u8],
    weight: &mut [u16],
    donor: &[u8],
    image_size: usize,
    variant: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) {
    for y in y0..y0 + height {
        for x in x0..x0 + width {
            let (sx, sy) = transform_coords(image_size, x, y, variant);
            let index = pixel_index(image_size, x, y);
            let source = u16::from(donor[pixel_index(image_size, sx, sy)]);
            let current = u16::from(image[index]) * weight[index];
            let next_weight = weight[index].saturating_add(1);
            image[index] =
                u8::try_from((current.saturating_add(source)) / next_weight).unwrap_or(u8::MAX);
            weight[index] = next_weight;
        }
    }
}

fn stamp_patch(
    image: &mut [u8],
    donor: &[u8],
    image_size: usize,
    variant: usize,
    x0: usize,
    y0: usize,
    patch: usize,
    rng: &mut Rng,
) {
    let y1 = (y0 + patch).min(image_size);
    let x1 = (x0 + patch).min(image_size);
    for y in y0..y1 {
        for x in x0..x1 {
            let (sx, sy) = transform_coords(image_size, x, y, variant);
            let source = donor[pixel_index(image_size, sx, sy)];
            if source > 96 && rng.chance(2, 3) {
                let index = pixel_index(image_size, x, y);
                image[index] = image[index].max(source.saturating_sub(24));
            }
        }
    }
}

fn clean_prior_image<'a>(
    prior: &'a CleanPrior,
    image_bytes: usize,
    image_index: usize,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let start = image_index
        .checked_mul(image_bytes)
        .ok_or("prior offset overflow")?;
    let end = start
        .checked_add(image_bytes)
        .ok_or("prior offset overflow")?;
    Ok(&prior.images[start..end])
}

fn transformed_prior_image(
    source: &[u8],
    image_size: usize,
    variant: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let mut out = vec![0_u8; image_bytes];
    for y in 0..image_size {
        for x in 0..image_size {
            let (sx, sy) = transform_coords(image_size, x, y, variant);
            out[pixel_index(image_size, x, y)] = source[pixel_index(image_size, sx, sy)];
        }
    }
    Ok(out)
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

fn shift_bitmap(
    input: &[u8],
    image_size: usize,
    shift_x: i32,
    shift_y: i32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let mut out = vec![0_u8; image_bytes];
    let size_i32 = i32::try_from(image_size)?;
    for y in 0..image_size {
        for x in 0..image_size {
            let sx = i32::try_from(x)? - shift_x;
            let sy = i32::try_from(y)? - shift_y;
            if sx < 0 || sy < 0 || sx >= size_i32 || sy >= size_i32 {
                continue;
            }
            let source_index = pixel_index(image_size, usize::try_from(sx)?, usize::try_from(sy)?);
            out[pixel_index(image_size, x, y)] = input[source_index];
        }
    }
    Ok(out)
}

fn overlay_prior(image: &mut [u8], other: &[u8], rng: &mut Rng) {
    for (target, source) in image.iter_mut().zip(other.iter()) {
        if *source > 96 && rng.chance(1, 5) {
            *target = (*target).max(source.saturating_sub(48));
        }
    }
}

fn splice_prior_rectangles(
    image: &mut [u8],
    image_size: usize,
    rng: &mut Rng,
    prior: &CleanPrior,
) -> Result<(), Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    let patch_count = 1 + usize::try_from(rng.rand_bounded(3))?;
    for _ in 0..patch_count {
        let donor_index = usize::try_from(rng.rand_bounded(u32::try_from(prior.image_count)?))?;
        let donor = clean_prior_image(prior, image_bytes, donor_index)?;
        let variant = usize::try_from(rng.rand_bounded(8))?;
        let donor_image = transformed_prior_image(donor, image_size, variant)?;
        let patch = 14 + usize::try_from(rng.rand_bounded(23))?;
        let max_start = image_size.saturating_sub(patch);
        let x0 = usize::try_from(rng.rand_bounded(u32::try_from(max_start + 1)?))?;
        let y0 = usize::try_from(rng.rand_bounded(u32::try_from(max_start + 1)?))?;
        let y1 = (y0 + patch).min(image_size);
        let x1 = (x0 + patch).min(image_size);
        for y in y0..y1 {
            for x in x0..x1 {
                let index = pixel_index(image_size, x, y);
                let source = donor_image[index];
                if source > 96 && rng.chance(2, 3) {
                    image[index] = source;
                } else if rng.chance(1, 5) {
                    image[index] = image[index].saturating_sub(40);
                }
            }
        }
    }
    Ok(())
}

fn dilate_bitmap(input: &[u8], image_size: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for y in 0..image_size {
        for x in 0..image_size {
            let value = input[pixel_index(image_size, x, y)];
            if value <= 96 {
                continue;
            }
            for dy in 0..3 {
                for dx in 0..3 {
                    let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
                    let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
                    if let (Some(nx), Some(ny)) = (nx, ny) {
                        if nx < image_size && ny < image_size {
                            let index = pixel_index(image_size, nx, ny);
                            out[index] = out[index].max(value.saturating_sub(32));
                        }
                    }
                }
            }
        }
    }
    out
}

fn jitter_prior(image: &mut [u8], rng: &mut Rng) {
    for value in image {
        if *value > 0 {
            if rng.chance(1, 14) {
                *value = 0;
            } else if rng.chance(1, 9) {
                *value = value.saturating_sub(64);
            } else if rng.chance(1, 18) {
                *value = value.saturating_add(24);
            }
        } else if rng.chance(1, 160) {
            *value = 120;
        }
    }
}

fn jitter_coordinate_prior(image: &mut [u8], rng: &mut Rng) {
    for value in image {
        if *value > 0 {
            if rng.chance(1, 20) {
                *value = 0;
            } else if rng.chance(1, 12) {
                *value = value.saturating_sub(48);
            }
        } else if rng.chance(1, 384) {
            *value = 96;
        }
    }
}

fn jitter_patch_prior(image: &mut [u8], rng: &mut Rng) {
    for value in image {
        if *value > 0 {
            if rng.chance(1, 18) {
                *value = 0;
            } else if rng.chance(1, 10) {
                *value = value.saturating_sub(48);
            } else if rng.chance(1, 20) {
                *value = value.saturating_add(20);
            }
        } else if rng.chance(1, 220) {
            *value = 104;
        }
    }
}

fn score_sample(image: &[u8], image_size: usize) -> Result<i64, Box<dyn std::error::Error>> {
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

    let mut ink_sum = 0_i64;
    let mut outer_ring_sum = 0_i64;
    let mut inner_ring_sum = 0_i64;
    let mut center_sum = 0_i64;
    let mut border_ink = 0_i64;
    let mut edge_count = 0_i64;

    for y in 0..image_size {
        for x in 0..image_size {
            let value = i64::from(image[pixel_index(image_size, x, y)]);
            ink_sum = ink_sum.saturating_add(value);
            if value > 64 {
                if x < 3 || y < 3 || x + 3 >= image_size || y + 3 >= image_size {
                    border_ink = border_ink.saturating_add(1);
                }
                let dx = i64::try_from(x)?.saturating_sub(center);
                let dy = i64::try_from(y)?.saturating_sub(center);
                let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
                if radius2 >= outer_low2 && radius2 <= outer_high2 {
                    outer_ring_sum = outer_ring_sum.saturating_add(value);
                }
                if radius2 >= inner_low2 && radius2 <= inner_high2 {
                    inner_ring_sum = inner_ring_sum.saturating_add(value);
                }
                if radius2 <= center_radius2 {
                    center_sum = center_sum.saturating_add(value);
                }
            }
            if x + 1 < image_size {
                let right = image[pixel_index(image_size, x + 1, y)];
                if abs_diff_u8(image[pixel_index(image_size, x, y)], right) > 64 {
                    edge_count = edge_count.saturating_add(1);
                }
            }
            if y + 1 < image_size {
                let down = image[pixel_index(image_size, x, y + 1)];
                if abs_diff_u8(image[pixel_index(image_size, x, y)], down) > 64 {
                    edge_count = edge_count.saturating_add(1);
                }
            }
        }
    }

    let target_ink_sum = i64::try_from(image_bytes)?.saturating_mul(47);
    let density_penalty = abs_i64(ink_sum.saturating_sub(target_ink_sum)) / 4;
    let score = outer_ring_sum
        .saturating_mul(4)
        .saturating_add(inner_ring_sum.saturating_mul(2))
        .saturating_add(center_sum)
        .saturating_add(edge_count.saturating_mul(18))
        .saturating_sub(border_ink.saturating_mul(450))
        .saturating_sub(density_penalty);
    Ok(score)
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
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!("sample has {} bytes, expected {image_bytes}", image.len()).into());
    }
    let mut sums = [0_u32; SIGNATURE_BINS];
    let mut counts = [0_u32; SIGNATURE_BINS];
    for y in 0..image_size {
        let bin_y = y * SIGNATURE_GRID / image_size;
        for x in 0..image_size {
            let bin_x = x * SIGNATURE_GRID / image_size;
            let bin = bin_y * SIGNATURE_GRID + bin_x;
            sums[bin] = sums[bin].saturating_add(u32::from(image[pixel_index(image_size, x, y)]));
            counts[bin] = counts[bin].saturating_add(1);
        }
    }
    let mut signature = [0_u16; SIGNATURE_BINS];
    for index in 0..SIGNATURE_BINS {
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
            selected_min_signature_distance: 0,
            selected_mean_signature_distance_q8: 0,
        });
    }

    let mut score_total = 0_i64;
    let mut selected_min_score = i64::MAX;
    let mut selected_max_score = i64::MIN;
    for candidate in selected {
        selected_min_score = selected_min_score.min(candidate.score);
        selected_max_score = selected_max_score.max(candidate.score);
        score_total = score_total.saturating_add(candidate.score);
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
        selected_min_signature_distance: min_distance,
        selected_mean_signature_distance_q8: mean_distance_q8,
    })
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

fn random_signed(rng: &mut Rng, magnitude: i32) -> Result<i32, Box<dyn std::error::Error>> {
    let span = u32::try_from(magnitude.saturating_mul(2).saturating_add(1))?;
    Ok(i32::try_from(rng.rand_bounded(span))? - magnitude)
}

fn apply_model_condition(
    model: &SampleModel,
    condition: usize,
    input: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match model {
        SampleModel::Conv3(model) => {
            let mut image = input.to_vec();
            for layer in &model.layers {
                image = apply_cv3_layer_condition(model, layer, condition, &image)?;
            }
            Ok(image)
        }
        SampleModel::Multichannel(model) => {
            let mut image = input.to_vec();
            for layer in &model.layers {
                image = apply_multichannel_layer_condition(model, layer, condition, &image)?;
            }
            Ok(image)
        }
    }
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
    let mut features = vec![0_i16; model.hidden_channels];
    for y in 0..model.image_size {
        for x in 0..model.image_size {
            let index = pixel_index(model.image_size, x, y);
            hidden_features(
                input,
                model.image_size,
                x,
                y,
                model.hidden_shift,
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

fn local_features(input: &[u8], image_size: usize, x: usize, y: usize, out: &mut [i16; KERNEL]) {
    let center = i16::from(input[pixel_index(image_size, x, y)]);
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
    for (channel, value) in out.iter_mut().enumerate() {
        let mut acc = 0_i64;
        for (weight, feature) in HIDDEN_KERNELS[channel].iter().zip(local.iter()) {
            acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
        }
        *value = signed_round_shift(acc, hidden_shift).clamp(-511, 511) as i16;
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

fn draw_circle(
    image: &mut [u8],
    image_size: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    thickness: i32,
    value: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    for offset in 0..thickness {
        let r = radius.saturating_sub(offset);
        if r <= 0 {
            continue;
        }
        let mut x = r;
        let mut y = 0_i32;
        let mut error = 1_i32 - x;
        while x >= y {
            plot_circle_points(image, image_size, cx, cy, x, y, value)?;
            y += 1;
            if error < 0 {
                error += 2 * y + 1;
            } else {
                x -= 1;
                error += 2 * (y - x) + 1;
            }
        }
    }
    Ok(())
}

fn plot_circle_points(
    image: &mut [u8],
    image_size: usize,
    cx: i32,
    cy: i32,
    x: i32,
    y: i32,
    value: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    for (px, py) in [
        (cx + x, cy + y),
        (cx + y, cy + x),
        (cx - y, cy + x),
        (cx - x, cy + y),
        (cx - x, cy - y),
        (cx - y, cy - x),
        (cx + y, cy - x),
        (cx + x, cy - y),
    ] {
        draw_point(image, image_size, px, py, 1, value)?;
    }
    Ok(())
}

fn draw_line(
    image: &mut [u8],
    image_size: usize,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    thickness: i32,
    value: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let dx = abs_i32(x1 - x0);
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -abs_i32(y1 - y0);
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        draw_point(image, image_size, x0, y0, thickness, value)?;
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = error.saturating_mul(2);
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
    Ok(())
}

fn draw_point(
    image: &mut [u8],
    image_size: usize,
    x: i32,
    y: i32,
    thickness: i32,
    value: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let radius = thickness.saturating_sub(1);
    for yy in y - radius..=y + radius {
        for xx in x - radius..=x + radius {
            if xx < 0 || yy < 0 {
                continue;
            }
            let ux = usize::try_from(xx)?;
            let uy = usize::try_from(yy)?;
            if ux >= image_size || uy >= image_size {
                continue;
            }
            let index = pixel_index(image_size, ux, uy);
            image[index] = image[index].max(value);
        }
    }
    Ok(())
}

fn abs_i32(value: i32) -> i32 {
    if value >= 0 {
        value
    } else {
        value.saturating_neg()
    }
}

fn abs_i64(value: i64) -> i64 {
    if value >= 0 {
        value
    } else {
        value.saturating_neg()
    }
}

fn abs_diff_u8(left: u8, right: u8) -> u8 {
    if left >= right {
        left - right
    } else {
        right - left
    }
}

fn abs_diff_u16(left: u16, right: u16) -> u16 {
    if left >= right {
        left - right
    } else {
        right - left
    }
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

    fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.rand_bounded(denominator) < numerator
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
    prior: Option<&CleanPrior>,
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
    let prior_path = prior
        .map(|prior| prior.path.display().to_string())
        .unwrap_or_default();
    let prior_count = prior.map(|prior| prior.image_count).unwrap_or(0);
    json_field(&mut out, "prior_clean", &prior_path, true);
    number_field(&mut out, "prior_clean_count", prior_count, true);
    number_field(&mut out, "image_size", model.image_size(), true);
    number_field(&mut out, "timesteps", model.timesteps(), true);
    number_field(&mut out, "layers", model.layer_count(), true);
    number_field(&mut out, "samples", config.sample_count, true);
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
