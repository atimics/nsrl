#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "nsrl.bitmap_denoise_multichannel_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLMCH\n";
const TEXT_MODEL_MAGIC: &[u8; 8] = b"NSRLTCH\n";
const KERNEL: usize = 9;
const HIDDEN_CHANNELS: usize = 8;
const POSITION_FEATURE_CHANNELS: usize = 6;
const TEXT_FEATURE_CHANNELS: usize = 16;
const TEXT_SIGNATURE_GRID: usize = 16;
const TEXT_SIGNATURE_BINS: usize = TEXT_SIGNATURE_GRID * TEXT_SIGNATURE_GRID;
const INK_THRESHOLD: u8 = 64;
const STRONG_INK_THRESHOLD: u8 = 160;
const TARGET_CLEAN_EDGE_CLEAR: usize = 3;
const TARGET_CLEAN_RADIUS_MARGIN: usize = 2;
const LAYOUT_INK_FLOOR: u16 = 32;
const LAYOUT_INK_MIDPOINT: u16 = 54;
const LAYOUT_INK_CEILING: u16 = 96;
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
type AuxTargets = (Option<Vec<u8>>, Option<Vec<u8>>);
type LayerGradientSums = (Vec<Vec<i64>>, Vec<i64>, u64);

#[derive(Debug, Clone)]
struct Config {
    dataset_root: PathBuf,
    out_dir: PathBuf,
    model_out: PathBuf,
    text_index_path: Option<PathBuf>,
    image_size: usize,
    timesteps: usize,
    layers: usize,
    epochs: usize,
    hidden_shift: u8,
    output_shift: u8,
    learning_shift: u8,
    bias_learning_shift: u8,
    max_weight_delta: i16,
    max_bias_delta: i16,
    aux_clean_target_weight: u16,
    aux_clean_target_mode: AuxCleanTargetMode,
    preview_pairs: usize,
}

impl Default for Config {
    fn default() -> Self {
        let dataset_root = PathBuf::from("data/processed/key-solomon-goetia-denoise-v1");
        let out_dir = dataset_root.join("baseline-multichannel-conv");
        let model_out = out_dir.join("model.nsrlmch");
        Self {
            dataset_root,
            out_dir,
            model_out,
            text_index_path: None,
            image_size: 128,
            timesteps: 8,
            layers: 3,
            epochs: 8,
            hidden_shift: 1,
            output_shift: 9,
            learning_shift: 24,
            bias_learning_shift: 30,
            max_weight_delta: 4,
            max_bias_delta: 12,
            aux_clean_target_weight: 0,
            aux_clean_target_mode: AuxCleanTargetMode::WholeImage,
            preview_pairs: 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuxCleanTargetMode {
    WholeImage,
    SuppressArtifacts,
    StrictArtifacts,
    SignedArtifacts,
}

impl AuxCleanTargetMode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "whole-image" => Ok(Self::WholeImage),
            "suppress-artifacts" => Ok(Self::SuppressArtifacts),
            "strict-artifacts" => Ok(Self::StrictArtifacts),
            "signed-artifacts" => Ok(Self::SignedArtifacts),
            _ => Err(format!(
                "unknown aux clean target mode: {value}; expected whole-image, suppress-artifacts, strict-artifacts, or signed-artifacts"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::WholeImage => "whole-image",
            Self::SuppressArtifacts => "suppress-artifacts",
            Self::StrictArtifacts => "strict-artifacts",
            Self::SignedArtifacts => "signed-artifacts",
        }
    }
}

#[derive(Debug, Clone)]
struct PairRow {
    corruption: String,
    timestep: usize,
    slice_id: String,
    text_signature: [u16; TEXT_SIGNATURE_BINS],
    text_signature_stats: SignatureStats,
}

#[derive(Debug, Clone, Copy)]
struct SignatureStats {
    global_mean: i16,
    row_means: [i16; TEXT_SIGNATURE_GRID],
    col_means: [i16; TEXT_SIGNATURE_GRID],
}

#[derive(Debug)]
struct SplitData {
    input: Vec<u8>,
    target: Vec<u8>,
    aux_target: Option<Vec<u8>>,
    aux_mask: Option<Vec<u8>>,
    rows: Vec<PairRow>,
}

#[derive(Debug)]
struct TextIndex {
    slice_ids: Vec<String>,
}

impl TextIndex {
    fn has_slice_id(&self, slice_id: &str) -> bool {
        self.slice_ids.iter().any(|known| known == slice_id)
    }
}

#[derive(Debug)]
struct ConvLayer {
    condition_weights: Vec<Vec<i8>>,
    condition_biases: Vec<i16>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct LayeredConvModel {
    image_size: usize,
    timesteps: usize,
    hidden_shift: u8,
    output_shift: u8,
    feature_count: usize,
    text_conditioned: bool,
    layers: Vec<ConvLayer>,
}

#[derive(Debug, Clone)]
struct EpochTrace {
    layer: usize,
    epoch: usize,
    train_raw_mean_abs_q8: u64,
    weight_min: i8,
    weight_max: i8,
    active_weight_count: usize,
    bias_min: i16,
    bias_max: i16,
}

#[derive(Debug)]
struct Metrics {
    pair_count: usize,
    pixel_count: u64,
    input_abs_error: u64,
    layer_abs_errors: Vec<u64>,
    predicted_abs_error: u64,
    input_aux_abs_error: u64,
    predicted_aux_abs_error: u64,
    input_mean_abs_q8: u64,
    layer_mean_abs_q8: Vec<u64>,
    predicted_mean_abs_q8: u64,
    input_aux_mean_abs_q8: u64,
    predicted_aux_mean_abs_q8: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-bitmap-multichannel-denoise: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.image_size == 0 || config.timesteps == 0 || config.epochs == 0 || config.layers == 0 {
        return Err("image size, timesteps, layers, and epochs must be positive".into());
    }
    if config.aux_clean_target_weight > u16::from(u8::MAX) + 1 {
        return Err("--aux-clean-target-weight must be <= 256".into());
    }
    fs::create_dir_all(&config.out_dir)?;

    let text_index = if let Some(path) = config.text_index_path.as_ref() {
        Some(read_text_index(path)?)
    } else {
        None
    };
    let train = read_split(&config, "train", text_index.as_ref())?;
    let eval = read_split(&config, "eval", text_index.as_ref())?;
    let (model, epochs) = train_model(&config, &train)?;
    write_model(&config.model_out, &model)?;

    let train_metrics = evaluate_split(&config, &model, "train", &train, config.preview_pairs)?;
    let eval_metrics = evaluate_split(&config, &model, "eval", &eval, config.preview_pairs)?;
    write_trace(&config, &model, &epochs, &train_metrics, &eval_metrics)?;

    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"train_predicted_mae\":\"{}\",\"eval_predicted_mae\":\"{}\",\"eval_input_copy_mae\":\"{}\",\"aux_clean_target_weight\":{},\"aux_clean_target_mode\":\"{}\",\"eval_aux_predicted_mae\":\"{}\",\"eval_aux_input_copy_mae\":\"{}\"}}",
        SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        format_q8(train_metrics.predicted_mean_abs_q8),
        format_q8(eval_metrics.predicted_mean_abs_q8),
        format_q8(eval_metrics.input_mean_abs_q8),
        config.aux_clean_target_weight,
        config.aux_clean_target_mode.as_str(),
        format_q8(eval_metrics.predicted_aux_mean_abs_q8),
        format_q8(eval_metrics.input_aux_mean_abs_q8),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-bitmap-multichannel-denoise [--dataset PATH] [--out-dir PATH] [--model-out PATH] [--text-index PATH] [--image-size N] [--timesteps N] [--layers N] [--epochs N] [--hidden-shift N] [--output-shift N] [--learning-shift N] [--bias-learning-shift N] [--max-weight-delta N] [--max-bias-delta N] [--aux-clean-target-weight N<=256] [--aux-clean-target-mode whole-image|suppress-artifacts|strict-artifacts|signed-artifacts] [--preview-pairs N]"
    );
}

fn parse_args<I>(mut args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let default = Config::default();
    let mut config = default.clone();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--dataset" | "--dataset-root" => {
                config.dataset_root = PathBuf::from(args.next().ok_or("--dataset requires PATH")?);
                if config.out_dir == default.out_dir {
                    config.out_dir = config.dataset_root.join("baseline-multichannel-conv");
                    config.model_out = config.out_dir.join("model.nsrlmch");
                }
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
                if config.model_out == default.model_out {
                    config.model_out = config.out_dir.join("model.nsrlmch");
                }
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--text-index" => {
                config.text_index_path = Some(PathBuf::from(
                    args.next().ok_or("--text-index requires PATH")?,
                ));
                if config.out_dir == default.out_dir {
                    config.out_dir = config.dataset_root.join("text-multichannel-conv");
                    config.model_out = config.out_dir.join("model.nsrltch");
                } else if config.model_out == default.model_out {
                    config.model_out = config.out_dir.join("model.nsrltch");
                }
            }
            "--image-size" => {
                config.image_size = args.next().ok_or("--image-size requires N")?.parse()?
            }
            "--timesteps" => {
                config.timesteps = args.next().ok_or("--timesteps requires N")?.parse()?
            }
            "--layers" => config.layers = args.next().ok_or("--layers requires N")?.parse()?,
            "--epochs" => config.epochs = args.next().ok_or("--epochs requires N")?.parse()?,
            "--hidden-shift" => {
                config.hidden_shift = args.next().ok_or("--hidden-shift requires N")?.parse()?
            }
            "--output-shift" => {
                config.output_shift = args.next().ok_or("--output-shift requires N")?.parse()?
            }
            "--learning-shift" => {
                config.learning_shift = args.next().ok_or("--learning-shift requires N")?.parse()?
            }
            "--bias-learning-shift" => {
                config.bias_learning_shift = args
                    .next()
                    .ok_or("--bias-learning-shift requires N")?
                    .parse()?
            }
            "--max-weight-delta" => {
                config.max_weight_delta = args
                    .next()
                    .ok_or("--max-weight-delta requires N")?
                    .parse()?
            }
            "--max-bias-delta" => {
                config.max_bias_delta = args.next().ok_or("--max-bias-delta requires N")?.parse()?
            }
            "--aux-clean-target-weight" => {
                config.aux_clean_target_weight = args
                    .next()
                    .ok_or("--aux-clean-target-weight requires N")?
                    .parse()?
            }
            "--aux-clean-target-mode" => {
                config.aux_clean_target_mode = AuxCleanTargetMode::parse(
                    &args.next().ok_or("--aux-clean-target-mode requires MODE")?,
                )?
            }
            "--preview-pairs" => {
                config.preview_pairs = args.next().ok_or("--preview-pairs requires N")?.parse()?
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(config)
}

fn read_split(
    config: &Config,
    split: &str,
    text_index: Option<&TextIndex>,
) -> Result<SplitData, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let input_path = config
        .dataset_root
        .join("pairs")
        .join(format!("{split}.input.ink{}.u8", config.image_size));
    let target_path = config
        .dataset_root
        .join("pairs")
        .join(format!("{split}.target.ink{}.u8", config.image_size));
    let rows_path = config
        .dataset_root
        .join("rows")
        .join(format!("{split}.pairs.jsonl"));
    let input = fs::read(&input_path)?;
    let target = fs::read(&target_path)?;
    if input.len() != target.len() {
        return Err(format!(
            "{split} input and target byte counts differ: {} vs {}",
            input.len(),
            target.len()
        )
        .into());
    }
    if input.len() % image_bytes != 0 {
        return Err(format!(
            "{split} byte count {} is not a multiple of image bytes {image_bytes}",
            input.len()
        )
        .into());
    }
    let rows = read_pair_rows(&rows_path)?;
    let pair_count = input.len() / image_bytes;
    if rows.len() != pair_count {
        return Err(format!(
            "{split} row count {} does not match pair count {pair_count}",
            rows.len()
        )
        .into());
    }
    if let Some(index) = text_index {
        let mut filtered_input = Vec::new();
        let mut filtered_target = Vec::new();
        let mut filtered_rows = Vec::new();
        for (pair_index, row) in rows.into_iter().enumerate() {
            if !index.has_slice_id(&row.slice_id) {
                continue;
            }
            let start = pair_index * image_bytes;
            let end = start + image_bytes;
            let mut row = row;
            row.text_signature = image_signature(&target[start..end], config.image_size)?;
            row.text_signature_stats = signature_stats(&row.text_signature);
            filtered_input.extend_from_slice(&input[start..end]);
            filtered_target.extend_from_slice(&target[start..end]);
            filtered_rows.push(row);
        }
        if filtered_rows.is_empty() {
            return Err(format!("{split} has no rows matching --text-index").into());
        }
        let (aux_target, aux_mask) = aux_targets_for_pairs(config, &filtered_target)?;
        return Ok(SplitData {
            input: filtered_input,
            target: filtered_target,
            aux_target,
            aux_mask,
            rows: filtered_rows,
        });
    }
    let (aux_target, aux_mask) = aux_targets_for_pairs(config, &target)?;
    Ok(SplitData {
        input,
        target,
        aux_target,
        aux_mask,
        rows,
    })
}

fn read_pair_rows(path: &Path) -> Result<Vec<PairRow>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let corruption = json_string_field(&line, "corruption")
            .ok_or_else(|| format!("{}:{} missing corruption", path.display(), index + 1))?;
        let timestep = json_usize_field(&line, "timestep")
            .ok_or_else(|| format!("{}:{} missing timestep", path.display(), index + 1))?;
        let slice_id = json_string_field(&line, "slice_id")
            .ok_or_else(|| format!("{}:{} missing slice_id", path.display(), index + 1))?;
        rows.push(PairRow {
            corruption,
            timestep,
            slice_id,
            text_signature: [0; TEXT_SIGNATURE_BINS],
            text_signature_stats: empty_signature_stats(),
        });
    }
    Ok(rows)
}

fn read_text_index(path: &Path) -> Result<TextIndex, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut slice_ids = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            return Err(format!(
                "{} line {} has {} fields, expected 9",
                path.display(),
                line_index + 1,
                fields.len()
            )
            .into());
        }
        let slice_id = fields[3].to_string();
        if !slice_ids.iter().any(|known| known == &slice_id) {
            slice_ids.push(slice_id);
        }
    }
    if slice_ids.is_empty() {
        return Err(format!("{} has no text index rows", path.display()).into());
    }
    Ok(TextIndex { slice_ids })
}

fn aux_targets_for_pairs(
    config: &Config,
    target: &[u8],
) -> Result<AuxTargets, Box<dyn std::error::Error>> {
    if config.aux_clean_target_weight == 0 {
        return Ok((None, None));
    }
    let image_bytes = checked_image_bytes(config.image_size)?;
    if !target.len().is_multiple_of(image_bytes) {
        return Err("target byte count is not a multiple of image bytes".into());
    }
    let mut aux = Vec::with_capacity(target.len());
    let mut mask = if matches!(
        config.aux_clean_target_mode,
        AuxCleanTargetMode::SuppressArtifacts
            | AuxCleanTargetMode::StrictArtifacts
            | AuxCleanTargetMode::SignedArtifacts
    ) {
        Some(Vec::with_capacity(target.len()))
    } else {
        None
    };
    for image in target.chunks_exact(image_bytes) {
        let clean = clean_stroke_target(image, config.image_size)?;
        if let Some(mask) = mask.as_mut() {
            match config.aux_clean_target_mode {
                AuxCleanTargetMode::WholeImage => {}
                AuxCleanTargetMode::SuppressArtifacts => {
                    for (&raw, &cleaned) in image.iter().zip(clean.iter()) {
                        mask.push(u8::from(raw > 0 && cleaned == 0));
                    }
                }
                AuxCleanTargetMode::StrictArtifacts => {
                    mask.extend(strict_artifact_mask(image, config.image_size)?);
                }
                AuxCleanTargetMode::SignedArtifacts => {
                    mask.extend(strict_artifact_mask(image, config.image_size)?);
                }
            }
        }
        aux.extend_from_slice(&clean);
    }
    Ok((Some(aux), mask))
}

fn strict_artifact_mask(
    image: &[u8],
    image_size: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "strict artifact mask got {} bytes, expected {image_bytes}",
            image.len()
        )
        .into());
    }
    let center = i64::try_from(image_size / 2)?;
    let radius_margin = TARGET_CLEAN_RADIUS_MARGIN.min(image_size / 2);
    let seal_radius = i64::try_from(image_size / 2 - radius_margin)?;
    let seal_radius2 = seal_radius.saturating_mul(seal_radius);
    let mut mask = vec![0_u8; image_bytes];
    for y in 0..image_size {
        for x in 0..image_size {
            let index = pixel_index(image_size, x, y);
            let value = image[index];
            if value == 0 {
                continue;
            }
            let near_border = x < TARGET_CLEAN_EDGE_CLEAR
                || y < TARGET_CLEAN_EDGE_CLEAR
                || x + TARGET_CLEAN_EDGE_CLEAR >= image_size
                || y + TARGET_CLEAN_EDGE_CLEAR >= image_size;
            let dx = i64::try_from(x)?.saturating_sub(center);
            let dy = i64::try_from(y)?.saturating_sub(center);
            let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            let outside_seal = radius2 > seal_radius2;
            let neighbors = neighbor_ink_count(image, image_size, x, y);
            let weak_unsupported = value <= INK_THRESHOLD && neighbors <= 3;
            let mid_unsupported =
                value > INK_THRESHOLD && value <= STRONG_INK_THRESHOLD && neighbors <= 1;
            mask[index] =
                u8::from(near_border || outside_seal || weak_unsupported || mid_unsupported);
        }
    }
    Ok(mask)
}

fn clean_stroke_target(
    image: &[u8],
    image_size: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "clean target got {} bytes, expected {image_bytes}",
            image.len()
        )
        .into());
    }
    let center = i64::try_from(image_size / 2)?;
    let radius_margin = TARGET_CLEAN_RADIUS_MARGIN.min(image_size / 2);
    let seal_radius = i64::try_from(image_size / 2 - radius_margin)?;
    let seal_radius2 = seal_radius.saturating_mul(seal_radius);
    let mut out = vec![0_u8; image_bytes];
    for y in 0..image_size {
        for x in 0..image_size {
            if x < TARGET_CLEAN_EDGE_CLEAR
                || y < TARGET_CLEAN_EDGE_CLEAR
                || x + TARGET_CLEAN_EDGE_CLEAR >= image_size
                || y + TARGET_CLEAN_EDGE_CLEAR >= image_size
            {
                continue;
            }
            let dx = i64::try_from(x)?.saturating_sub(center);
            let dy = i64::try_from(y)?.saturating_sub(center);
            let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            if radius2 > seal_radius2 {
                continue;
            }
            let index = pixel_index(image_size, x, y);
            let value = image[index];
            let neighbors = neighbor_ink_count(image, image_size, x, y);
            out[index] = if value > STRONG_INK_THRESHOLD {
                value
            } else if value > INK_THRESHOLD && neighbors >= 2 {
                value.saturating_sub(16)
            } else if value > 40 && neighbors >= 4 {
                value.saturating_sub(40)
            } else {
                0
            };
        }
    }
    Ok(out)
}

fn train_model(
    config: &Config,
    train: &SplitData,
) -> Result<(LayeredConvModel, Vec<EpochTrace>), Box<dyn std::error::Error>> {
    let mut model = LayeredConvModel {
        image_size: config.image_size,
        timesteps: config.timesteps,
        hidden_shift: config.hidden_shift,
        output_shift: config.output_shift,
        feature_count: feature_count(config),
        text_conditioned: config.text_index_path.is_some(),
        layers: Vec::new(),
    };
    let mut traces = Vec::new();
    let mut layer_inputs = train.input.clone();
    for layer_index in 0..config.layers {
        let layer_number = layer_index + 1;
        let (mut layer, mut layer_traces) =
            train_layer(config, train, &layer_inputs, layer_number)?;
        tune_condition_blends(config, &mut layer, train, &layer_inputs)?;
        layer_inputs = apply_layer_to_split(config, &layer, train, &layer_inputs)?;
        model.layers.push(layer);
        traces.append(&mut layer_traces);
    }
    Ok((model, traces))
}

fn train_layer(
    config: &Config,
    train: &SplitData,
    layer_inputs: &[u8],
    layer_number: usize,
) -> Result<(ConvLayer, Vec<EpochTrace>), Box<dyn std::error::Error>> {
    let condition_count = condition_count(config)?;
    if layer_inputs.len() != train.target.len() {
        return Err("layer input and target byte counts differ".into());
    }
    let mut layer = ConvLayer {
        condition_weights: vec![vec![0; feature_count(config)]; condition_count],
        condition_biases: vec![0; condition_count],
        condition_blend_shifts: vec![u8::MAX; condition_count],
    };
    let mut traces = Vec::new();
    let image_bytes = checked_image_bytes(config.image_size)?;
    for epoch in 0..config.epochs {
        let pair_count = train.rows.len();
        let feature_dim = feature_count(config);
        // Per-pair-range gradient accumulation. Pure over the immutable `layer`
        // and dataset, so chunks run on separate threads and their partial i64
        // grads are summed deterministically (sums stay far below i64::MAX, so
        // saturating_add never saturates -> addition is associative -> the
        // parallel result is bit-identical to the serial path).
        let accumulate = |start: usize, end: usize| -> Result<LayerGradientSums, String> {
            let mut weight_grads = vec![vec![0_i64; feature_dim]; condition_count];
            let mut bias_grads = vec![0_i64; condition_count];
            let mut raw_error = 0_u64;
            for pair_index in start..end {
                let row = &train.rows[pair_index];
                let condition = condition_index(config, row).map_err(|e| e.to_string())?;
                let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
                let target =
                    &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
                let aux_target = train
                    .aux_target
                    .as_ref()
                    .map(|aux| &aux[pair_index * image_bytes..(pair_index + 1) * image_bytes]);
                let aux_mask = train
                    .aux_mask
                    .as_ref()
                    .map(|mask| &mask[pair_index * image_bytes..(pair_index + 1) * image_bytes]);
                for y in 0..config.image_size {
                    for x in 0..config.image_size {
                        let index = pixel_index(config.image_size, x, y);
                        let mut features = vec![0_i16; feature_dim];
                        conditioned_features(
                            input,
                            config.image_size,
                            x,
                            y,
                            config.hidden_shift,
                            row,
                            &mut features,
                        );
                        let predicted = predict_raw_pixel(
                            &layer,
                            config.output_shift,
                            condition,
                            input[index],
                            &features,
                        );
                        let error = training_error(
                            config,
                            predicted,
                            target[index],
                            aux_target.map(|aux| aux[index]),
                            aux_mask.map(|mask| mask[index]),
                        );
                        raw_error = raw_error.saturating_add(abs_i16(error));
                        for (channel, &feature) in features.iter().enumerate() {
                            weight_grads[condition][channel] = weight_grads[condition][channel]
                                .saturating_add(i64::from(error) * i64::from(feature));
                        }
                        bias_grads[condition] =
                            bias_grads[condition].saturating_add(i64::from(error));
                    }
                }
            }
            Ok((weight_grads, bias_grads, raw_error))
        };
        let worker_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(pair_count.max(1));
        let (weight_grads, bias_grads, raw_error) = if worker_count <= 1 {
            accumulate(0, pair_count)?
        } else {
            let accumulate_ref = &accumulate;
            let chunk = pair_count.div_ceil(worker_count);
            std::thread::scope(
                |scope| -> Result<LayerGradientSums, Box<dyn std::error::Error>> {
                    let mut handles = Vec::new();
                    let mut start = 0;
                    while start < pair_count {
                        let end = (start + chunk).min(pair_count);
                        handles.push(scope.spawn(move || accumulate_ref(start, end)));
                        start = end;
                    }
                    let mut weight_grads = vec![vec![0_i64; feature_dim]; condition_count];
                    let mut bias_grads = vec![0_i64; condition_count];
                    let mut raw_error = 0_u64;
                    for handle in handles {
                        let (w, b, r) = match handle.join() {
                            Ok(result) => result?,
                            Err(payload) => std::panic::resume_unwind(payload),
                        };
                        for condition in 0..condition_count {
                            for channel in 0..feature_dim {
                                weight_grads[condition][channel] = weight_grads[condition][channel]
                                    .saturating_add(w[condition][channel]);
                            }
                            bias_grads[condition] =
                                bias_grads[condition].saturating_add(b[condition]);
                        }
                        raw_error = raw_error.saturating_add(r);
                    }
                    Ok((weight_grads, bias_grads, raw_error))
                },
            )?
        };
        for (weights, grads) in layer.condition_weights.iter_mut().zip(weight_grads.iter()) {
            for (weight, grad) in weights.iter_mut().zip(grads.iter()) {
                let delta = signed_round_shift(*grad, config.learning_shift).clamp(
                    -i64::from(config.max_weight_delta),
                    i64::from(config.max_weight_delta),
                );
                let next = i16::from(*weight)
                    .saturating_add(delta as i16)
                    .clamp(-127, 127);
                *weight = next as i8;
            }
        }
        for (bias, grad) in layer.condition_biases.iter_mut().zip(bias_grads.iter()) {
            let delta = signed_round_shift(*grad, config.bias_learning_shift).clamp(
                -i64::from(config.max_bias_delta),
                i64::from(config.max_bias_delta),
            );
            *bias = bias.saturating_add(delta as i16);
        }
        traces.push(EpochTrace {
            layer: layer_number,
            epoch: epoch + 1,
            train_raw_mean_abs_q8: mean_q8(raw_error, train_pixel_count(config, train)?),
            weight_min: layer_weight_min(&layer),
            weight_max: layer_weight_max(&layer),
            active_weight_count: layer_active_weight_count(&layer),
            bias_min: *layer.condition_biases.iter().min().unwrap_or(&0),
            bias_max: *layer.condition_biases.iter().max().unwrap_or(&0),
        });
    }
    Ok((layer, traces))
}

fn tune_condition_blends(
    config: &Config,
    layer: &mut ConvLayer,
    train: &SplitData,
    layer_inputs: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let condition_count = condition_count(config)?;
    let candidates = [u8::MAX, 0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut errors = vec![0_u64; condition_count * candidates.len()];
    let image_bytes = checked_image_bytes(config.image_size)?;
    for pair_index in 0..train.rows.len() {
        let row = &train.rows[pair_index];
        let condition = condition_index(config, row)?;
        let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let aux_target = train
            .aux_target
            .as_ref()
            .map(|aux| &aux[pair_index * image_bytes..(pair_index + 1) * image_bytes]);
        let aux_mask = train
            .aux_mask
            .as_ref()
            .map(|mask| &mask[pair_index * image_bytes..(pair_index + 1) * image_bytes]);
        for y in 0..config.image_size {
            for x in 0..config.image_size {
                let index = pixel_index(config.image_size, x, y);
                let mut features = vec![0_i16; feature_count(config)];
                conditioned_features(
                    input,
                    config.image_size,
                    x,
                    y,
                    config.hidden_shift,
                    row,
                    &mut features,
                );
                let raw = predict_raw_pixel(
                    layer,
                    config.output_shift,
                    condition,
                    input[index],
                    &features,
                );
                for (candidate_index, &candidate) in candidates.iter().enumerate() {
                    let predicted = blend_pixel(input[index], raw, candidate);
                    let offset = condition * candidates.len() + candidate_index;
                    errors[offset] = errors[offset].saturating_add(training_pixel_loss(
                        config,
                        predicted,
                        target[index],
                        aux_target.map(|aux| aux[index]),
                        aux_mask.map(|mask| mask[index]),
                    ));
                }
            }
        }
    }
    layer
        .condition_blend_shifts
        .resize(condition_count, u8::MAX);
    for condition in 0..condition_count {
        let mut best_index = 0_usize;
        let mut best_error = u64::MAX;
        for (candidate_index, _) in candidates.iter().enumerate() {
            let error = errors[condition * candidates.len() + candidate_index];
            if error < best_error {
                best_error = error;
                best_index = candidate_index;
            }
        }
        layer.condition_blend_shifts[condition] = candidates[best_index];
    }
    Ok(())
}

fn apply_layer_to_split(
    config: &Config,
    layer: &ConvLayer,
    data: &SplitData,
    layer_inputs: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut out = vec![0_u8; layer_inputs.len()];
    for pair_index in 0..data.rows.len() {
        let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let prediction = predict_layer_image(config, layer, &data.rows[pair_index], input)?;
        out[pair_index * image_bytes..(pair_index + 1) * image_bytes].copy_from_slice(&prediction);
    }
    Ok(out)
}

fn evaluate_split(
    config: &Config,
    model: &LayeredConvModel,
    split: &str,
    data: &SplitData,
    preview_pairs: usize,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut input_abs_error = 0_u64;
    let mut layer_abs_errors = vec![0_u64; model.layers.len()];
    let mut predicted_abs_error = 0_u64;
    let mut input_aux_abs_error = 0_u64;
    let mut predicted_aux_abs_error = 0_u64;
    let mut preview = Vec::new();
    for pair_index in 0..data.rows.len() {
        let input = &data.input[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &data.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let aux_target = data
            .aux_target
            .as_ref()
            .map(|aux| &aux[pair_index * image_bytes..(pair_index + 1) * image_bytes]);
        let mut prediction = input.to_vec();
        for (layer_index, layer) in model.layers.iter().enumerate() {
            prediction = predict_layer_image(config, layer, &data.rows[pair_index], &prediction)?;
            for index in 0..image_bytes {
                layer_abs_errors[layer_index] = layer_abs_errors[layer_index]
                    .saturating_add(abs_diff_u8(prediction[index], target[index]));
            }
        }
        for index in 0..image_bytes {
            input_abs_error =
                input_abs_error.saturating_add(abs_diff_u8(input[index], target[index]));
            predicted_abs_error =
                predicted_abs_error.saturating_add(abs_diff_u8(prediction[index], target[index]));
            if let Some(aux_target) = aux_target {
                input_aux_abs_error = input_aux_abs_error
                    .saturating_add(abs_diff_u8(input[index], aux_target[index]));
                predicted_aux_abs_error = predicted_aux_abs_error
                    .saturating_add(abs_diff_u8(prediction[index], aux_target[index]));
            }
        }
        if preview.len() < preview_pairs {
            preview.push((input.to_vec(), prediction, target.to_vec()));
        }
    }
    write_preview(
        &config
            .out_dir
            .join(format!("{split}.input-pred-target.pgm")),
        config.image_size,
        &preview,
    )?;
    let pixel_count = train_pixel_count(config, data)?;
    let layer_mean_abs_q8 = layer_abs_errors
        .iter()
        .map(|&error| mean_q8(error, pixel_count))
        .collect();
    Ok(Metrics {
        pair_count: data.rows.len(),
        pixel_count,
        input_abs_error,
        layer_abs_errors,
        predicted_abs_error,
        input_aux_abs_error,
        predicted_aux_abs_error,
        input_mean_abs_q8: mean_q8(input_abs_error, pixel_count),
        layer_mean_abs_q8,
        predicted_mean_abs_q8: mean_q8(predicted_abs_error, pixel_count),
        input_aux_mean_abs_q8: mean_q8(input_aux_abs_error, pixel_count),
        predicted_aux_mean_abs_q8: mean_q8(predicted_aux_abs_error, pixel_count),
    })
}

fn predict_layer_image(
    config: &Config,
    layer: &ConvLayer,
    row: &PairRow,
    input: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let condition = condition_index(config, row)?;
    let blend_shift = layer
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut prediction = vec![0_u8; input.len()];
    for y in 0..config.image_size {
        for x in 0..config.image_size {
            let index = pixel_index(config.image_size, x, y);
            let mut features = vec![0_i16; feature_count(config)];
            conditioned_features(
                input,
                config.image_size,
                x,
                y,
                config.hidden_shift,
                row,
                &mut features,
            );
            let raw = predict_raw_pixel(
                layer,
                config.output_shift,
                condition,
                input[index],
                &features,
            );
            prediction[index] = blend_pixel(input[index], raw, blend_shift);
        }
    }
    Ok(prediction)
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

fn conditioned_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    hidden_shift: u8,
    row: &PairRow,
    out: &mut [i16],
) {
    let image_channels = out.len().min(HIDDEN_CHANNELS);
    hidden_features(
        input,
        image_size,
        x,
        y,
        hidden_shift,
        &mut out[..image_channels],
    );
    let mut offset = HIDDEN_CHANNELS;
    if out.len() >= offset + POSITION_FEATURE_CHANNELS {
        position_features(
            image_size,
            x,
            y,
            &mut out[offset..offset + POSITION_FEATURE_CHANNELS],
        );
        offset += POSITION_FEATURE_CHANNELS;
    }
    if out.len() < offset + TEXT_FEATURE_CHANNELS {
        return;
    }
    let input_center = i16::from(input[pixel_index(image_size, x, y)]);
    text_signature_features(
        &row.text_signature,
        &row.text_signature_stats,
        image_size,
        x,
        y,
        input_center,
        &mut out[offset..offset + TEXT_FEATURE_CHANNELS],
    );
}

fn interpolated_text_signature(
    signature: &[u16; TEXT_SIGNATURE_BINS],
    image_size: usize,
    x: usize,
    y: usize,
) -> i16 {
    if image_size <= 1 {
        return i16::try_from(signature[0].min(255)).unwrap_or(0);
    }
    let grid_max = TEXT_SIGNATURE_GRID - 1;
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
    let at = |xx: usize, yy: usize| -> i64 {
        i64::from(signature[yy * TEXT_SIGNATURE_GRID + xx].min(255))
    };
    let weighted = at(x0, y0)
        .saturating_mul(ix)
        .saturating_mul(iy)
        .saturating_add(at(x1, y0).saturating_mul(wx).saturating_mul(iy))
        .saturating_add(at(x0, y1).saturating_mul(ix).saturating_mul(wy))
        .saturating_add(at(x1, y1).saturating_mul(wx).saturating_mul(wy));
    i16::try_from((weighted + 32_768) >> 16).unwrap_or(0)
}

fn text_signature_features(
    signature: &[u16; TEXT_SIGNATURE_BINS],
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
    let step = (image_size / TEXT_SIGNATURE_GRID.max(1)).max(1);
    let center = interpolated_text_signature(signature, image_size, x, y);
    let left = interpolated_text_signature(signature, image_size, x.saturating_sub(step), y);
    let right = interpolated_text_signature(
        signature,
        image_size,
        (x + step).min(image_size.saturating_sub(1)),
        y,
    );
    let up = interpolated_text_signature(signature, image_size, x, y.saturating_sub(step));
    let down = interpolated_text_signature(
        signature,
        image_size,
        x,
        (y + step).min(image_size.saturating_sub(1)),
    );
    let up_left = interpolated_text_signature(
        signature,
        image_size,
        x.saturating_sub(step),
        y.saturating_sub(step),
    );
    let down_right = interpolated_text_signature(
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
        .saturating_mul(TEXT_SIGNATURE_GRID)
        .checked_div(image_size)
        .unwrap_or(0)
        .min(TEXT_SIGNATURE_GRID - 1)
}

fn empty_signature_stats() -> SignatureStats {
    SignatureStats {
        global_mean: 0,
        row_means: [0; TEXT_SIGNATURE_GRID],
        col_means: [0; TEXT_SIGNATURE_GRID],
    }
}

fn signature_stats(signature: &[u16; TEXT_SIGNATURE_BINS]) -> SignatureStats {
    let global_mean = signature_global_mean(signature);
    let mut row_means = [0_i16; TEXT_SIGNATURE_GRID];
    let mut col_means = [0_i16; TEXT_SIGNATURE_GRID];
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

fn signature_global_mean(signature: &[u16; TEXT_SIGNATURE_BINS]) -> i16 {
    let total: u32 = signature
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_BINS).unwrap_or(1)).unwrap_or(0)
}

fn signature_row_mean(signature: &[u16; TEXT_SIGNATURE_BINS], row: usize) -> i16 {
    let row = row.min(TEXT_SIGNATURE_GRID - 1);
    let start = row * TEXT_SIGNATURE_GRID;
    let total: u32 = signature[start..start + TEXT_SIGNATURE_GRID]
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
}

fn signature_col_mean(signature: &[u16; TEXT_SIGNATURE_BINS], col: usize) -> i16 {
    let col = col.min(TEXT_SIGNATURE_GRID - 1);
    let mut total = 0_u32;
    for row in 0..TEXT_SIGNATURE_GRID {
        total = total.saturating_add(u32::from(
            signature[row * TEXT_SIGNATURE_GRID + col].min(255),
        ));
    }
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
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

fn predict_raw_pixel(
    layer: &ConvLayer,
    output_shift: u8,
    condition: usize,
    input_center: u8,
    features: &[i16],
) -> u8 {
    let mut acc = i64::from(*layer.condition_biases.get(condition).unwrap_or(&0));
    let weights = layer
        .condition_weights
        .get(condition)
        .cloned()
        .unwrap_or_else(|| vec![0; features.len()]);
    for (weight, feature) in weights.iter().zip(features.iter()) {
        acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
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

fn condition_count(config: &Config) -> Result<usize, Box<dyn std::error::Error>> {
    CORRUPTION_KINDS
        .len()
        .checked_mul(config.timesteps)
        .ok_or_else(|| "condition count overflow".into())
}

fn feature_count(config: &Config) -> usize {
    if config.text_index_path.is_some() {
        HIDDEN_CHANNELS + POSITION_FEATURE_CHANNELS + TEXT_FEATURE_CHANNELS
    } else {
        HIDDEN_CHANNELS + POSITION_FEATURE_CHANNELS
    }
}

fn condition_index(config: &Config, row: &PairRow) -> Result<usize, Box<dyn std::error::Error>> {
    if row.timestep == 0 || row.timestep > config.timesteps {
        return Err(format!(
            "timestep {} is outside 1..={}",
            row.timestep, config.timesteps
        )
        .into());
    }
    let corruption = CORRUPTION_KINDS
        .iter()
        .position(|&kind| kind == row.corruption)
        .ok_or_else(|| format!("unknown corruption kind: {}", row.corruption))?;
    Ok(corruption * config.timesteps + (row.timestep - 1))
}

fn pixel_index(image_size: usize, x: usize, y: usize) -> usize {
    y * image_size + x
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

fn checked_image_bytes(image_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image byte count overflow".into())
}

fn image_signature(
    image: &[u8],
    image_size: usize,
) -> Result<[u16; TEXT_SIGNATURE_BINS], Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "image signature got {} bytes, expected {image_bytes}",
            image.len()
        )
        .into());
    }
    let mut sums = [0_u32; TEXT_SIGNATURE_BINS];
    let mut counts = [0_u32; TEXT_SIGNATURE_BINS];
    for y in 0..image_size {
        let bin_y = y * TEXT_SIGNATURE_GRID / image_size;
        for x in 0..image_size {
            let bin_x = x * TEXT_SIGNATURE_GRID / image_size;
            let bin = bin_y * TEXT_SIGNATURE_GRID + bin_x;
            sums[bin] = sums[bin].saturating_add(u32::from(image[pixel_index(image_size, x, y)]));
            counts[bin] = counts[bin].saturating_add(1);
        }
    }
    let mut signature = [0_u16; TEXT_SIGNATURE_BINS];
    for index in 0..TEXT_SIGNATURE_BINS {
        if counts[index] != 0 {
            let mean =
                u16::try_from(sums[index].saturating_add(counts[index] / 2) / counts[index])?;
            signature[index] = sharpen_signature_value(mean);
        }
    }
    Ok(signature)
}

fn sharpen_signature_value(value: u16) -> u16 {
    if value <= LAYOUT_INK_FLOOR {
        return 0;
    }
    if value >= LAYOUT_INK_CEILING {
        return 255;
    }
    if value <= LAYOUT_INK_MIDPOINT {
        return ((u32::from(value - LAYOUT_INK_FLOOR) * 96)
            / u32::from(LAYOUT_INK_MIDPOINT - LAYOUT_INK_FLOOR).max(1)) as u16;
    }
    let strong = 96
        + ((u32::from(value - LAYOUT_INK_MIDPOINT) * 159)
            / u32::from(LAYOUT_INK_CEILING - LAYOUT_INK_MIDPOINT).max(1));
    strong.min(255) as u16
}

fn train_pixel_count(config: &Config, data: &SplitData) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::try_from(data.rows.len())?
        .saturating_mul(u64::try_from(checked_image_bytes(config.image_size)?)?))
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

fn abs_i16(value: i16) -> u64 {
    if value >= 0 {
        u64::from(value as u16)
    } else {
        u64::from(value.unsigned_abs())
    }
}

fn training_error(
    config: &Config,
    predicted: u8,
    target: u8,
    aux_target: Option<u8>,
    aux_mask: Option<u8>,
) -> i16 {
    if config.aux_clean_target_mode == AuxCleanTargetMode::SignedArtifacts {
        let target = signed_artifact_target(config, target, aux_target, aux_mask);
        return (i32::from(target) - i32::from(predicted))
            .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    let base = i32::from(target) - i32::from(predicted);
    let aux = if aux_mask.is_some_and(|mask| mask == 0) {
        0
    } else {
        aux_target
            .map(|aux| {
                let aux_error = i32::from(aux) - i32::from(predicted);
                signed_round_shift(
                    i64::from(aux_error).saturating_mul(i64::from(config.aux_clean_target_weight)),
                    8,
                ) as i32
            })
            .unwrap_or(0)
    };
    base.saturating_add(aux)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn training_pixel_loss(
    config: &Config,
    predicted: u8,
    target: u8,
    aux_target: Option<u8>,
    aux_mask: Option<u8>,
) -> u64 {
    if config.aux_clean_target_mode == AuxCleanTargetMode::SignedArtifacts {
        return abs_diff_u8(
            predicted,
            signed_artifact_target(config, target, aux_target, aux_mask),
        );
    }
    let base = abs_diff_u8(predicted, target);
    let aux = if aux_mask.is_some_and(|mask| mask == 0) {
        0
    } else {
        aux_target
            .map(|aux| {
                ((u128::from(abs_diff_u8(predicted, aux))
                    * u128::from(config.aux_clean_target_weight))
                    / 256_u128) as u64
            })
            .unwrap_or(0)
    };
    base.saturating_add(aux)
}

fn signed_artifact_target(
    config: &Config,
    target: u8,
    aux_target: Option<u8>,
    aux_mask: Option<u8>,
) -> u8 {
    if aux_mask.is_some_and(|mask| mask == 0) {
        return target;
    }
    let Some(aux_target) = aux_target else {
        return target;
    };
    let delta = i64::from(i32::from(aux_target) - i32::from(target))
        .saturating_mul(i64::from(config.aux_clean_target_weight));
    let shifted = signed_round_shift(delta, 8);
    let target = i64::from(target).saturating_add(shifted).clamp(0, 255);
    target as u8
}

fn abs_diff_u8(left: u8, right: u8) -> u64 {
    if left >= right {
        u64::from(left - right)
    } else {
        u64::from(right - left)
    }
}

fn mean_q8(total: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    ((u128::from(total) * 256_u128) / u128::from(count)) as u64
}

fn format_q8(value: u64) -> String {
    let mut whole = value / 256;
    let mut fraction = ((value % 256) * 100 + 128) / 256;
    if fraction == 100 {
        whole = whole.saturating_add(1);
        fraction = 0;
    }
    format!("{whole}.{fraction:02}")
}

fn layer_weight_min(layer: &ConvLayer) -> i8 {
    layer
        .condition_weights
        .iter()
        .flat_map(|weights| weights.iter())
        .copied()
        .min()
        .unwrap_or(0)
}

fn layer_weight_max(layer: &ConvLayer) -> i8 {
    layer
        .condition_weights
        .iter()
        .flat_map(|weights| weights.iter())
        .copied()
        .max()
        .unwrap_or(0)
}

fn layer_active_weight_count(layer: &ConvLayer) -> usize {
    layer
        .condition_weights
        .iter()
        .flat_map(|weights| weights.iter())
        .filter(|&&weight| weight != 0)
        .count()
}

fn write_model(path: &Path, model: &LayeredConvModel) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    if model.text_conditioned {
        bytes.extend_from_slice(TEXT_MODEL_MAGIC);
    } else {
        bytes.extend_from_slice(MODEL_MAGIC);
    }
    bytes.extend_from_slice(&checked_u32(model.image_size, "image_size")?.to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.timesteps, "timesteps")?.to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.hidden_shift).to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.output_shift).to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.feature_count, "feature count")?.to_le_bytes());
    bytes
        .extend_from_slice(&checked_u32(CORRUPTION_KINDS.len(), "corruption count")?.to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.layers.len(), "layer count")?.to_le_bytes());
    for layer in &model.layers {
        for weights in &layer.condition_weights {
            for &weight in weights {
                bytes.push(weight as u8);
            }
        }
        for bias in &layer.condition_biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        bytes.extend_from_slice(&layer.condition_blend_shifts);
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| format!("{label} exceeds u32").into())
}

fn write_trace(
    config: &Config,
    model: &LayeredConvModel,
    epochs: &[EpochTrace],
    train: &Metrics,
    eval: &Metrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str("{\n");
    json_field(&mut out, "schema", SCHEMA, true);
    json_field(
        &mut out,
        "dataset_root",
        &config.dataset_root.display().to_string(),
        true,
    );
    json_field(
        &mut out,
        "model_out",
        &config.model_out.display().to_string(),
        true,
    );
    let text_index = config
        .text_index_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    json_field(&mut out, "text_index", &text_index, true);
    number_field(&mut out, "feature_count", model.feature_count, true);
    number_field(
        &mut out,
        "text_feature_channels",
        if model.text_conditioned {
            TEXT_FEATURE_CHANNELS
        } else {
            0
        },
        true,
    );
    number_field(&mut out, "image_size", config.image_size, true);
    number_field(&mut out, "timesteps", config.timesteps, true);
    number_field(&mut out, "layers", model.layers.len(), true);
    number_field(&mut out, "epochs", config.epochs, true);
    number_field(&mut out, "hidden_channels", HIDDEN_CHANNELS, true);
    number_field(
        &mut out,
        "hidden_shift",
        usize::from(config.hidden_shift),
        true,
    );
    number_field(
        &mut out,
        "output_shift",
        usize::from(config.output_shift),
        true,
    );
    number_field(
        &mut out,
        "learning_shift",
        usize::from(config.learning_shift),
        true,
    );
    number_field(
        &mut out,
        "aux_clean_target_weight",
        usize::from(config.aux_clean_target_weight),
        true,
    );
    json_field(
        &mut out,
        "aux_clean_target_mode",
        config.aux_clean_target_mode.as_str(),
        true,
    );
    out.push_str("  \"layer_models\":[");
    for (layer_index, layer) in model.layers.iter().enumerate() {
        if layer_index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"layer\":{},\"weight_min\":{},\"weight_max\":{},\"active_weight_count\":{},\"condition_blend_shifts\":[",
            layer_index + 1,
            layer_weight_min(layer),
            layer_weight_max(layer),
            layer_active_weight_count(layer),
        ));
        for (index, shift) in layer.condition_blend_shifts.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            if *shift == u8::MAX {
                out.push_str("\"copy\"");
            } else {
                out.push_str(&shift.to_string());
            }
        }
        out.push_str("]}");
    }
    out.push_str("],\n");
    out.push_str("  \"epochs_trace\":[");
    for (index, epoch) in epochs.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"layer\":{},\"epoch\":{},\"train_raw_mean_abs\":\"{}\",\"weight_min\":{},\"weight_max\":{},\"active_weight_count\":{},\"bias_min\":{},\"bias_max\":{}}}",
            epoch.layer,
            epoch.epoch,
            format_q8(epoch.train_raw_mean_abs_q8),
            epoch.weight_min,
            epoch.weight_max,
            epoch.active_weight_count,
            epoch.bias_min,
            epoch.bias_max,
        ));
    }
    out.push_str("],\n");
    metrics_field(&mut out, "train", train, true);
    metrics_field(&mut out, "eval", eval, false);
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

fn metrics_field(out: &mut String, key: &str, metrics: &Metrics, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":{");
    out.push_str(&format!(
        "\"pair_count\":{},\"pixel_count\":{},\"input_abs_error\":{},\"input_aux_abs_error\":{},",
        metrics.pair_count,
        metrics.pixel_count,
        metrics.input_abs_error,
        metrics.input_aux_abs_error,
    ));
    out.push_str("\"layer_abs_errors\":[");
    for (index, error) in metrics.layer_abs_errors.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&error.to_string());
    }
    out.push_str("],");
    out.push_str(&format!(
        "\"predicted_abs_error\":{},\"predicted_aux_abs_error\":{},\"input_mean_abs_q8\":{},\"input_aux_mean_abs_q8\":{},",
        metrics.predicted_abs_error,
        metrics.predicted_aux_abs_error,
        metrics.input_mean_abs_q8,
        metrics.input_aux_mean_abs_q8,
    ));
    out.push_str("\"layer_mean_abs_q8\":[");
    for (index, value) in metrics.layer_mean_abs_q8.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push_str("],\"layer_mean_abs\":[");
    for (index, value) in metrics.layer_mean_abs_q8.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&format_q8(*value));
        out.push('"');
    }
    out.push_str(&format!(
        "],\"predicted_mean_abs_q8\":{},\"predicted_aux_mean_abs_q8\":{},\"input_mean_abs\":\"{}\",\"predicted_mean_abs\":\"{}\",\"input_aux_mean_abs\":\"{}\",\"predicted_aux_mean_abs\":\"{}\"",
        metrics.predicted_mean_abs_q8,
        metrics.predicted_aux_mean_abs_q8,
        format_q8(metrics.input_mean_abs_q8),
        format_q8(metrics.predicted_mean_abs_q8),
        format_q8(metrics.input_aux_mean_abs_q8),
        format_q8(metrics.predicted_aux_mean_abs_q8),
    ));
    out.push('}');
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

fn json_string_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\":\"");
    let start = line.find(&needle)? + needle.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in line[start..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_usize_field(line: &str, field: &str) -> Option<usize> {
    let needle = format!("\"{field}\":");
    let start = line.find(&needle)? + needle.len();
    let mut end = start;
    let bytes = line.as_bytes();
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    line[start..end].parse().ok()
}

fn write_preview(
    path: &Path,
    image_size: usize,
    triples: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Result<(), Box<dyn std::error::Error>> {
    if triples.is_empty() {
        return Ok(());
    }
    let columns = 2_usize;
    let gap = 4_usize;
    let triple_width = image_size * 3 + gap * 2;
    let rows = triples.len().div_ceil(columns);
    let width = columns * triple_width + (columns + 1) * gap;
    let height = rows * image_size + (rows + 1) * gap;
    let mut sheet = vec![255_u8; width * height];
    for (triple_index, (input, prediction, target)) in triples.iter().enumerate() {
        let col = triple_index % columns;
        let row = triple_index / columns;
        let x0 = gap + col * (triple_width + gap);
        let y0 = gap + row * (image_size + gap);
        for (side, image) in [input, prediction, target].iter().enumerate() {
            let side_x = x0 + side * (image_size + gap);
            for y in 0..image_size {
                for x in 0..image_size {
                    let value = 255_u8.saturating_sub(image[pixel_index(image_size, x, y)]);
                    sheet[(y0 + y) * width + side_x + x] = value;
                }
            }
        }
    }
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{width} {height}\n255\n")?;
    file.write_all(&sheet)?;
    Ok(())
}
